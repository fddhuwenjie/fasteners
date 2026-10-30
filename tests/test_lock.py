# -*- coding: utf-8 -*-

#    Copyright (C) 2014 Yahoo! Inc. All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import collections
from concurrent import futures
import random
import threading
import time

import pytest

import fasteners
from fasteners import _utils

# NOTE(harlowja): Sleep a little so now() can not be the same (which will
# cause false positives when our overlap detection code runs). If there are
# real overlaps then they will still exist.
NAPPY_TIME = 0.05

# We will spend this amount of time doing some "fake" work.
WORK_TIMES = [(0.01 + x / 100.0) for x in range(0, 5)]

# If latches/events take longer than this to become empty/set, something is
# usually wrong and should be debugged instead of deadlocking...
WAIT_TIMEOUT = 300

THREAD_COUNT = 20


def _find_overlaps(times, start, end):
    overlaps = 0
    for (s, e) in times:
        if s >= start and e <= end:
            overlaps += 1
    return overlaps


def _spawn_variation(readers, writers, max_workers=None):
    start_stops = collections.deque()
    lock = fasteners.ReaderWriterLock()

    def read_func(ident):
        with lock.read_lock():
            enter_time = time.monotonic()
            time.sleep(WORK_TIMES[ident % len(WORK_TIMES)])
            exit_time = time.monotonic()
            start_stops.append((lock.READER, enter_time, exit_time))
            time.sleep(NAPPY_TIME)

    def write_func(ident):
        with lock.write_lock():
            enter_time = time.monotonic()
            time.sleep(WORK_TIMES[ident % len(WORK_TIMES)])
            exit_time = time.monotonic()
            start_stops.append((lock.WRITER, enter_time, exit_time))
            time.sleep(NAPPY_TIME)

    if max_workers is None:
        max_workers = max(0, readers) + max(0, writers)
    if max_workers > 0:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as e:
            count = 0
            for _i in range(0, readers):
                e.submit(read_func, count)
                count += 1
            for _i in range(0, writers):
                e.submit(write_func, count)
                count += 1

    writer_times = []
    reader_times = []
    for (lock_type, start, stop) in list(start_stops):
        if lock_type == lock.WRITER:
            writer_times.append((start, stop))
        else:
            reader_times.append((start, stop))
    return writer_times, reader_times


def _daemon_thread(target):
    t = threading.Thread(target=target)
    t.daemon = True
    return t


def _wait_for(predicate, timeout=WAIT_TIMEOUT, interval=0.001):
    """Poll ``predicate`` until it is true or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


@pytest.mark.parametrize("contextmanager", [True, False])
def test_no_double_writers(contextmanager):
    lock = fasteners.ReaderWriterLock()
    watch = _utils.StopWatch(duration=5)
    watch.start()
    dups = collections.deque()
    active = collections.deque()

    def acquire_check_ctx(me):
        with lock.write_lock():
            if len(active) >= 1:
                dups.append(me)
                dups.extend(active)
            active.append(me)
            time.sleep(random.random() / 100)
            active.remove(me)

    def acquire_check_plain(me):
        lock.acquire_write_lock()
        if len(active) >= 1:
            dups.append(me)
            dups.extend(active)
        active.append(me)
        time.sleep(random.random() / 100)
        active.remove(me)
        lock.release_write_lock()

    def run():
        me = threading.current_thread()
        while not watch.expired():
            if contextmanager:
                acquire_check_ctx(me)
            else:
                acquire_check_plain(me)

    threads = []
    for i in range(0, THREAD_COUNT):
        t = _daemon_thread(run)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()

    assert not dups
    assert not active


@pytest.mark.parametrize("contextmanager", [True, False])
def test_no_concurrent_readers_writers(contextmanager):
    lock = fasteners.ReaderWriterLock()
    watch = _utils.StopWatch(duration=5)
    watch.start()
    dups = collections.deque()
    active = collections.deque()

    def acquire_check_ctx(me, reader):
        if reader:
            lock_func = lock.read_lock
        else:
            lock_func = lock.write_lock
        with lock_func():
            if not reader:
                # There should be no-one else currently active, if there
                # is ensure we capture them so that we can later blow-up
                # the test.
                if len(active) >= 1:
                    dups.append(me)
                    dups.extend(active)
            active.append(me)
            time.sleep(random.random() / 100)
            active.remove(me)

    def acquire_check_plain(me, reader):
        if reader:
            lock_func, unlock_func = lock.acquire_read_lock, lock.release_read_lock
        else:
            lock_func, unlock_func = lock.acquire_write_lock, lock.release_write_lock

        lock_func()
        if not reader:
            # There should be no-one else currently active, if there
            # is ensure we capture them so that we can later blow-up
            # the test.
            if len(active) >= 1:
                dups.append(me)
                dups.extend(active)
        active.append(me)
        time.sleep(random.random() / 100)
        active.remove(me)
        unlock_func()

    def run():
        me = threading.current_thread()
        while not watch.expired():
            if contextmanager:
                acquire_check_ctx(me, random.choice([True, False]))
            else:
                acquire_check_plain(me, random.choice([True, False]))

    threads = []
    for i in range(0, THREAD_COUNT):
        t = _daemon_thread(run)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()

    assert not dups
    assert not active


def test_writer_abort():
    lock = fasteners.ReaderWriterLock()
    assert lock.owner is None

    with pytest.raises(RuntimeError):
        with lock.write_lock():
            assert lock.owner == lock.WRITER
            raise RuntimeError("Broken")

    assert lock.owner is None


def test_reader_abort():
    lock = fasteners.ReaderWriterLock()
    assert lock.owner is None

    with pytest.raises(RuntimeError):
        with lock.read_lock():
            assert lock.owner == lock.READER
            raise RuntimeError("Broken")

    assert lock.owner is None


def test_double_reader_abort():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def double_bad_reader():
        with lock.read_lock():
            with lock.read_lock():
                raise RuntimeError("Broken")

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)

    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(double_bad_reader)
            else:
                e.submit(happy_writer)

    assert sum(a == 'w' for a in activated) == 10


def test_double_reader_writer():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()
    active = threading.Event()

    def double_reader():
        with lock.read_lock():
            active.set()
            while not lock.has_pending_writers:
                time.sleep(0.001)
            with lock.read_lock():
                activated.append(lock.owner)

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)

    reader = _daemon_thread(double_reader)
    reader.start()
    active.wait(WAIT_TIMEOUT)
    assert active.is_set()

    writer = _daemon_thread(happy_writer)
    writer.start()

    reader.join()
    writer.join()
    assert list(activated) == ['r', 'w']


def test_reader_chaotic():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def chaotic_reader(blow_up):
        with lock.read_lock():
            if blow_up:
                raise RuntimeError("Broken")
            else:
                activated.append(lock.owner)

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)

    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(chaotic_reader, blow_up=bool(i % 4 == 0))
            else:
                e.submit(happy_writer)

    assert sum(a == 'w' for a in activated) == 10
    assert sum(a == 'r' for a in activated) == 5


def test_writer_chaotic():
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def chaotic_writer(blow_up):
        with lock.write_lock():
            if blow_up:
                raise RuntimeError("Broken")
            else:
                activated.append(lock.owner)

    def happy_reader():
        with lock.read_lock():
            activated.append(lock.owner)

    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(chaotic_writer, blow_up=bool(i % 4 == 0))
            else:
                e.submit(happy_reader)

    assert sum(a == 'w' for a in activated) == 5
    assert sum(a == 'r' for a in activated) == 10


def test_writer_reader_writer_ctx():
    lock = fasteners.ReaderWriterLock()
    with lock.write_lock():
        assert lock.is_writer()
        with lock.read_lock():
            assert lock.is_reader()
            with lock.write_lock():
                assert lock.is_writer()


def test_writer_reader_writer_plain():
    lock = fasteners.ReaderWriterLock()
    lock.acquire_write_lock()
    assert lock.is_writer()
    lock.acquire_read_lock()
    assert lock.is_reader()
    lock.acquire_write_lock()
    assert lock.is_writer()


def test_single_reader_writer_ctx():
    lock = fasteners.ReaderWriterLock()
    with lock.read_lock():
        assert lock.is_reader()
    with lock.write_lock():
        assert lock.is_writer()
    with lock.read_lock():
        assert lock.is_reader()
    assert not lock.is_reader()
    assert not lock.is_writer()


def test_single_reader_writer_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    lock.release_read_lock()

    lock.acquire_write_lock()
    assert lock.is_writer()
    lock.release_write_lock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    lock.release_read_lock()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_reader_to_writer_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.read_lock():
        with pytest.raises(RuntimeError):
            with lock.write_lock():
                pass
        assert lock.is_reader()
        assert not lock.is_writer()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_reader_to_writer_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    with pytest.raises(RuntimeError):
        lock.acquire_write_lock()
    assert lock.is_reader()
    assert not lock.is_writer()
    lock.release_read_lock()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_writer_to_reader_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.write_lock():
        with lock.read_lock():
            assert lock.is_writer()
            assert lock.is_reader()

        assert lock.is_writer()
        assert not lock.is_reader()

    assert not lock.is_writer()
    assert not lock.is_reader()


def test_writer_to_reader_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_write_lock()
    lock.acquire_read_lock()
    assert lock.is_writer()
    assert lock.is_reader()

    lock.release_read_lock()
    assert lock.is_writer()
    assert not lock.is_reader()

    lock.release_write_lock()
    assert not lock.is_writer()
    assert not lock.is_reader()


def test_double_writer_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.write_lock():
        assert not lock.is_reader()
        assert lock.is_writer()

        with lock.write_lock():
            assert lock.is_writer()

        assert lock.is_writer()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_double_writer_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_write_lock()
    assert not lock.is_reader()
    assert lock.is_writer()

    lock.acquire_write_lock()
    assert lock.is_writer()

    lock.release_write_lock()
    assert lock.is_writer()

    lock.release_write_lock()
    assert not lock.is_reader()
    assert not lock.is_writer()


def test_double_reader_ctx():
    lock = fasteners.ReaderWriterLock()

    with lock.read_lock():
        assert lock.is_reader()
        assert not lock.is_writer()

        with lock.read_lock():
            assert lock.is_reader()

        assert lock.is_reader()

    assert not lock.is_reader()
    assert not lock.is_writer()


def test_double_reader_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    assert not lock.is_writer()

    lock.acquire_read_lock()
    assert lock.is_reader()

    lock.release_read_lock()
    assert lock.is_reader()

    lock.release_read_lock()
    assert not lock.is_reader()
    assert not lock.is_writer()


def test_multi_reader_multi_writer():
    writer_times, reader_times = _spawn_variation(10, 10)
    assert len(writer_times) == 10
    assert len(reader_times) == 10

    for (start, stop) in writer_times:
        assert _find_overlaps(reader_times, start, stop) == 0
        assert _find_overlaps(writer_times, start, stop) == 1
    for (start, stop) in reader_times:
        assert _find_overlaps(writer_times, start, stop) == 0


def test_multi_reader_single_writer():
    writer_times, reader_times = _spawn_variation(9, 1)
    assert len(writer_times) == 1
    assert len(reader_times) == 9

    start, stop = writer_times[0]
    assert _find_overlaps(reader_times, start, stop) == 0


def test_multi_writer():
    writer_times, reader_times = _spawn_variation(0, 10)
    assert len(writer_times) == 10
    assert len(reader_times) == 0

    for (start, stop) in writer_times:
        assert _find_overlaps(writer_times, start, stop) == 1


def test_deadlock_reproducer():
    lock = fasteners.ReaderWriterLock()

    def thread1():
        for _ in range(0, 1000):
            print("thread1 iteration: {}".format(_))
            print("thread1 waiting for write lock")
            with lock.write_lock():
                print("thread1 got write lock")
                print("thread1 waiting for read lock")
                with lock.read_lock():
                    print("thread1 got read lock")
                print("thread1 released read lock")
            print("thread1 released write lock")

    def thread2():
        for _ in range(0, 1000):
            print("thread2 iteration: {}".format(_))
            print("thread2 waiting for write lock")
            with lock.write_lock():
                print("thread2 got write lock")
            print("thread2 released write lock")

    funcs = (thread1, thread2)

    threads = []
    for func in funcs:
        t = _daemon_thread(func)
        threads.append(t)
        t.start()
    while threads:
        t = threads.pop()
        t.join()


def test_error_when_releasing_write_lock_without_write_lock():
    lock = fasteners.ReaderWriterLock()
    with pytest.raises(RuntimeError):
        lock.release_write_lock()


def test_error_when_releasing_read_lock_without_read_lock():
    lock = fasteners.ReaderWriterLock()
    with pytest.raises(RuntimeError):
        lock.release_read_lock()


# ---------------------------------------------------------------------------
# read -> write upgrade / write -> read downgrade
# ---------------------------------------------------------------------------


def test_upgrade_downgrade_plain():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    assert lock.is_reader()
    assert lock.owner == lock.READER

    assert lock.upgrade_read_lock() is True
    assert lock.is_writer()
    assert not lock.is_reader()
    assert lock.owner == lock.WRITER

    lock.downgrade_write_lock()
    assert lock.is_reader()
    assert not lock.is_writer()
    assert lock.owner == lock.READER

    lock.release_read_lock()
    assert not lock.is_reader()
    assert not lock.is_writer()
    assert lock.owner is None


def test_upgrade_downgrade_inside_read_context():
    lock = fasteners.ReaderWriterLock()

    with lock.read_lock():
        assert lock.is_reader()
        assert lock.upgrade_read_lock() is True
        assert lock.is_writer(check_pending=False)
        lock.downgrade_write_lock()
        assert lock.is_reader()
        assert not lock.is_writer()

    assert lock.owner is None


def test_repeated_upgrade_downgrade_cycles():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    for _ in range(0, 5):
        assert lock.upgrade_read_lock() is True
        assert lock.is_writer()
        lock.downgrade_write_lock()
        assert lock.is_reader()
    lock.release_read_lock()

    assert lock.owner is None


def test_upgraded_write_can_be_released_instead_of_downgraded():
    lock = fasteners.ReaderWriterLock()

    with lock.read_lock():
        assert lock.upgrade_read_lock() is True
        assert lock.is_writer()
        # the converted read lock is gone, so releasing the write unlocks
        # the lock completely.
        lock.release_write_lock()
        assert lock.owner is None
        assert not lock.is_reader()

    assert lock.owner is None


def test_upgraded_writer_is_exclusive():
    lock = fasteners.ReaderWriterLock()
    entered = collections.deque()

    def reader_or_writer(kind, event):
        if kind == 'r':
            lock.acquire_read_lock()
        else:
            lock.acquire_write_lock()
        entered.append(kind)
        event.set()
        if kind == 'r':
            lock.release_read_lock()
        else:
            lock.release_write_lock()

    lock.acquire_read_lock()
    assert lock.upgrade_read_lock() is True

    reader_event = threading.Event()
    writer_event = threading.Event()
    reader_thread = _daemon_thread(
        lambda: reader_or_writer('r', reader_event))
    writer_thread = _daemon_thread(
        lambda: reader_or_writer('w', writer_event))
    reader_thread.start()
    writer_thread.start()

    # nobody can enter while the upgraded write lock is held
    assert not reader_event.wait(0.05)
    assert not writer_event.wait(0.05)
    assert not entered

    lock.downgrade_write_lock()
    # the queued writer keeps waiting (the caller atomically became a
    # reader), and writer preference means that *new* reader can not overtake
    # that queued writer either.
    assert not reader_event.wait(0.05)
    assert not writer_event.wait(0.05)
    assert not entered

    lock.release_read_lock()
    # writer first (FIFO), reader only once the writer is done
    assert writer_event.wait(WAIT_TIMEOUT)
    assert reader_event.wait(WAIT_TIMEOUT)
    reader_thread.join(WAIT_TIMEOUT)
    writer_thread.join(WAIT_TIMEOUT)
    assert not reader_thread.is_alive()
    assert not writer_thread.is_alive()
    assert list(entered) == ['w', 'r']
    assert lock.owner is None


def test_downgrade_is_atomic_against_queued_writer():
    lock = fasteners.ReaderWriterLock()
    writer_got_lock = threading.Event()

    def queued_writer():
        lock.acquire_write_lock()
        writer_got_lock.set()
        lock.release_write_lock()

    lock.acquire_read_lock()
    assert lock.upgrade_read_lock() is True

    writer_thread = _daemon_thread(queued_writer)
    writer_thread.start()
    assert _wait_for(lambda: lock.has_pending_writers)
    assert not writer_got_lock.is_set()

    lock.downgrade_write_lock()
    # the conversion is atomic: right after it the caller must still be a
    # reader and the queued writer must not have slipped in.
    assert lock.is_reader()
    assert not lock.is_writer()
    assert lock.owner == lock.READER
    assert not writer_got_lock.wait(0.05)

    lock.release_read_lock()
    assert writer_got_lock.wait(WAIT_TIMEOUT)
    writer_thread.join(WAIT_TIMEOUT)
    assert not writer_thread.is_alive()
    assert lock.owner is None


def test_upgrade_without_read_lock_raises():
    lock = fasteners.ReaderWriterLock()

    with pytest.raises(RuntimeError):
        lock.upgrade_read_lock()
    assert lock.owner is None


def test_upgrade_with_nested_read_locks_raises():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    lock.acquire_read_lock()
    with pytest.raises(RuntimeError):
        lock.upgrade_read_lock()

    # state untouched: two read releases are still needed
    assert lock.is_reader()
    assert not lock.is_writer()
    lock.release_read_lock()
    assert lock.is_reader()
    lock.release_read_lock()
    assert lock.owner is None


def test_upgrade_by_writer_raises():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_write_lock()
    with pytest.raises(RuntimeError):
        lock.upgrade_read_lock()
    assert lock.is_writer()
    assert lock._writer_entries == 1
    lock.release_write_lock()
    assert lock.owner is None


def test_upgrade_negative_timeout_raises():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    with pytest.raises(ValueError):
        lock.upgrade_read_lock(timeout=-1)
    assert lock.is_reader()
    lock.release_read_lock()


def test_downgrade_without_write_lock_raises():
    lock = fasteners.ReaderWriterLock()

    with pytest.raises(RuntimeError):
        lock.downgrade_write_lock()

    lock.acquire_read_lock()
    with pytest.raises(RuntimeError):
        lock.downgrade_write_lock()
    assert lock.is_reader()
    assert not lock.is_writer()
    lock.release_read_lock()
    assert lock.owner is None


def test_downgrade_with_nested_write_locks_raises():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_write_lock()
    lock.acquire_write_lock()
    assert lock._writer_entries == 2
    with pytest.raises(RuntimeError):
        lock.downgrade_write_lock()

    assert lock.is_writer()
    assert lock._writer_entries == 2
    lock.release_write_lock()
    lock.release_write_lock()
    assert lock.owner is None


def test_downgrade_while_writer_also_reads_raises():
    lock = fasteners.ReaderWriterLock()

    with lock.write_lock():
        lock.acquire_read_lock()
        with pytest.raises(RuntimeError):
            lock.downgrade_write_lock()
        assert lock.is_writer()
        lock.release_read_lock()

    assert lock.owner is None


def test_upgraded_writer_reentrancy():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    assert lock.upgrade_read_lock() is True
    lock.acquire_write_lock()
    assert lock._writer_entries == 2
    assert lock.is_writer()
    lock.release_write_lock()
    assert lock._writer_entries == 1
    assert lock.is_writer()

    lock.downgrade_write_lock()
    assert lock.is_reader()
    lock.release_read_lock()
    assert lock.owner is None


def test_upgrade_waits_for_existing_reader():
    lock = fasteners.ReaderWriterLock()
    upgrader_ready = threading.Event()
    upgraded = threading.Event()
    result = {}

    def upgrader():
        lock.acquire_read_lock()
        upgrader_ready.set()
        result['ok'] = lock.upgrade_read_lock(timeout=WAIT_TIMEOUT)
        upgraded.set()
        lock.downgrade_write_lock()
        lock.release_read_lock()

    lock.acquire_read_lock()
    upgrader_thread = _daemon_thread(upgrader)
    upgrader_thread.start()

    assert upgrader_ready.wait(WAIT_TIMEOUT)
    assert _wait_for(lambda: lock.has_pending_writers)
    assert not upgraded.wait(0.05)

    lock.release_read_lock()
    assert upgraded.wait(WAIT_TIMEOUT)
    assert result['ok'] is True
    upgrader_thread.join(WAIT_TIMEOUT)
    assert not upgrader_thread.is_alive()
    assert lock.owner is None


def test_upgrade_timeout_zero_when_blocked():
    lock = fasteners.ReaderWriterLock()
    other_ready = threading.Event()
    release_other = threading.Event()

    def other_reader():
        lock.acquire_read_lock()
        other_ready.set()
        release_other.wait(WAIT_TIMEOUT)
        lock.release_read_lock()

    other_thread = _daemon_thread(other_reader)
    other_thread.start()
    assert other_ready.wait(WAIT_TIMEOUT)

    lock.acquire_read_lock()
    start = time.monotonic()
    assert lock.upgrade_read_lock(timeout=0) is False
    assert time.monotonic() - start < 0.5

    # the read lock is retained and no upgrade intent is left queued
    assert lock.is_reader()
    assert not lock.has_pending_writers
    release_other.set()
    other_thread.join(WAIT_TIMEOUT)
    assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT) is True
    lock.downgrade_write_lock()
    lock.release_read_lock()
    assert lock.owner is None


def test_upgrade_timeout_elapses_then_succeeds():
    lock = fasteners.ReaderWriterLock()
    other_ready = threading.Event()

    def slow_reader():
        lock.acquire_read_lock()
        other_ready.set()
        time.sleep(0.3)
        lock.release_read_lock()

    other_thread = _daemon_thread(slow_reader)
    other_thread.start()
    assert other_ready.wait(WAIT_TIMEOUT)

    lock.acquire_read_lock()
    start = time.monotonic()
    assert lock.upgrade_read_lock(timeout=0.1) is False
    elapsed = time.monotonic() - start
    assert 0.1 <= elapsed < 1.0

    # failed upgrade kept the read lock; retrying succeeds once readers drain
    assert lock.is_reader()
    assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT) is True
    lock.downgrade_write_lock()
    lock.release_read_lock()
    other_thread.join(WAIT_TIMEOUT)
    assert lock.owner is None


def test_upgrade_rejected_when_plain_writer_queued_ahead():
    lock = fasteners.ReaderWriterLock()
    writer_got_lock = threading.Event()

    def queued_writer():
        lock.acquire_write_lock()
        writer_got_lock.set()
        lock.release_write_lock()

    lock.acquire_read_lock()
    writer_thread = _daemon_thread(queued_writer)
    writer_thread.start()
    assert _wait_for(lambda: lock.has_pending_writers)

    # a waiting writer can not be overtaken; the call returns immediately
    # even with an unbounded timeout.
    start = time.monotonic()
    assert lock.upgrade_read_lock() is False
    assert time.monotonic() - start < 0.5
    assert lock.is_reader()
    assert not lock.is_writer()

    lock.release_read_lock()
    assert writer_got_lock.wait(WAIT_TIMEOUT)
    writer_thread.join(WAIT_TIMEOUT)
    assert lock.owner is None


def test_upgrade_rejected_when_another_upgrader_waiting():
    lock = fasteners.ReaderWriterLock()
    a_ready = threading.Event()
    b_ready = threading.Event()
    start_upgrade = threading.Event()
    release_b = threading.Event()
    a_upgraded = threading.Event()
    b_result = {}

    def reader_a():
        lock.acquire_read_lock()
        a_ready.set()
        assert start_upgrade.wait(WAIT_TIMEOUT)
        assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT) is True
        a_upgraded.set()
        time.sleep(0.05)
        lock.downgrade_write_lock()
        lock.release_read_lock()

    def reader_b():
        assert a_ready.wait(WAIT_TIMEOUT)
        lock.acquire_read_lock()
        b_ready.set()
        assert _wait_for(lambda: lock.has_pending_writers)
        b_result['ok'] = lock.upgrade_read_lock(timeout=0)
        assert lock.is_reader()
        release_b.wait(WAIT_TIMEOUT)
        lock.release_read_lock()

    thread_a = _daemon_thread(reader_a)
    thread_b = _daemon_thread(reader_b)
    thread_a.start()
    thread_b.start()

    # only let A upgrade once both readers are inside
    assert b_ready.wait(WAIT_TIMEOUT)
    start_upgrade.set()
    assert _wait_for(lambda: 'ok' in b_result)
    assert b_result['ok'] is False
    assert not a_upgraded.is_set()

    release_b.set()
    assert a_upgraded.wait(WAIT_TIMEOUT)
    thread_a.join(WAIT_TIMEOUT)
    thread_b.join(WAIT_TIMEOUT)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert lock.owner is None


def test_new_reader_cannot_overtake_waiting_upgrader():
    lock = fasteners.ReaderWriterLock()
    a_ready = threading.Event()
    b_ready = threading.Event()
    start_upgrade = threading.Event()
    a_upgraded = threading.Event()
    c_entered = threading.Event()

    def reader_a():
        lock.acquire_read_lock()
        a_ready.set()
        assert start_upgrade.wait(WAIT_TIMEOUT)
        assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT) is True
        a_upgraded.set()
        time.sleep(0.05)
        lock.downgrade_write_lock()
        lock.release_read_lock()

    def reader_b():
        assert a_ready.wait(WAIT_TIMEOUT)
        lock.acquire_read_lock()
        b_ready.set()
        time.sleep(0.2)
        lock.release_read_lock()

    def reader_c():
        # only attempt once the upgrade is queued
        assert _wait_for(lambda: lock.has_pending_writers)
        lock.acquire_read_lock()
        c_entered.set()
        lock.release_read_lock()

    thread_a = _daemon_thread(reader_a)
    thread_b = _daemon_thread(reader_b)
    thread_c = _daemon_thread(reader_c)
    thread_a.start()
    thread_b.start()
    assert b_ready.wait(WAIT_TIMEOUT)
    thread_c.start()

    # both readers are inside; now queue A's upgrade
    start_upgrade.set()
    assert a_upgraded.wait(WAIT_TIMEOUT)
    # C must not have been admitted ahead of the granted upgrade
    assert not c_entered.is_set()

    thread_a.join(WAIT_TIMEOUT)
    thread_b.join(WAIT_TIMEOUT)
    assert c_entered.wait(WAIT_TIMEOUT)
    thread_c.join(WAIT_TIMEOUT)
    for t in (thread_a, thread_b, thread_c):
        assert not t.is_alive()
    assert lock.owner is None


def test_writer_queued_behind_upgrader_runs_next():
    lock = fasteners.ReaderWriterLock()
    a_ready = threading.Event()
    b_ready = threading.Event()
    start_upgrade = threading.Event()
    order = collections.deque()

    def reader_a():
        lock.acquire_read_lock()
        a_ready.set()
        assert start_upgrade.wait(WAIT_TIMEOUT)
        assert lock.upgrade_read_lock(timeout=WAIT_TIMEOUT) is True
        order.append('upgrade')
        time.sleep(0.05)
        lock.downgrade_write_lock()
        lock.release_read_lock()

    def reader_b():
        assert a_ready.wait(WAIT_TIMEOUT)
        lock.acquire_read_lock()
        b_ready.set()
        time.sleep(0.15)
        lock.release_read_lock()

    def queued_writer():
        assert _wait_for(lambda: lock.has_pending_writers)
        lock.acquire_write_lock()
        order.append('writer')
        lock.release_write_lock()

    thread_a = _daemon_thread(reader_a)
    thread_b = _daemon_thread(reader_b)
    thread_w = _daemon_thread(queued_writer)
    thread_a.start()
    thread_b.start()
    assert b_ready.wait(WAIT_TIMEOUT)
    thread_w.start()
    start_upgrade.set()

    thread_a.join(WAIT_TIMEOUT)
    thread_b.join(WAIT_TIMEOUT)
    thread_w.join(WAIT_TIMEOUT)
    for t in (thread_a, thread_b, thread_w):
        assert not t.is_alive()
    assert list(order) == ['upgrade', 'writer']
    assert lock.owner is None


def test_failed_upgrade_keeps_read_lock_and_queue_usable():
    lock = fasteners.ReaderWriterLock()
    a_ready = threading.Event()
    a_done = threading.Event()
    writer_got_lock = threading.Event()
    a_result = {}

    def reader_a():
        lock.acquire_read_lock()
        a_ready.set()
        a_result['ok'] = lock.upgrade_read_lock(timeout=0.1)
        a_result['still_reader'] = lock.is_reader()
        a_done.set()
        lock.release_read_lock()

    def queued_writer():
        assert _wait_for(lambda: lock.has_pending_writers)
        lock.acquire_write_lock()
        writer_got_lock.set()
        lock.release_write_lock()

    # main thread is the other reader blocking the upgrade
    lock.acquire_read_lock()
    thread_a = _daemon_thread(reader_a)
    thread_w = _daemon_thread(queued_writer)
    thread_a.start()
    assert a_ready.wait(WAIT_TIMEOUT)
    thread_w.start()

    assert a_done.wait(WAIT_TIMEOUT)
    assert a_result['ok'] is False
    assert a_result['still_reader'] is True

    # upgrade intent cleaned up, but the plain writer still has to wait for
    # both readers to leave
    assert not writer_got_lock.is_set()
    time.sleep(0.2)
    lock.release_read_lock()
    assert writer_got_lock.wait(WAIT_TIMEOUT)

    thread_a.join(WAIT_TIMEOUT)
    thread_w.join(WAIT_TIMEOUT)
    assert not thread_a.is_alive()
    assert not thread_w.is_alive()
    assert not lock.has_pending_writers
    assert lock.owner is None


class _InterruptingCondition(threading.Condition):
    """Condition whose first blocking wait is interrupted."""

    def wait(self, timeout=None):
        raise KeyboardInterrupt


def test_upgrade_interrupted_cleans_up_intent():
    lock = fasteners.ReaderWriterLock(condition_cls=_InterruptingCondition)
    upgrader_ready = threading.Event()
    interrupted = threading.Event()
    still_reader = {}

    def upgrader():
        lock.acquire_read_lock()
        upgrader_ready.set()
        try:
            lock.upgrade_read_lock(timeout=None)
        except KeyboardInterrupt:
            interrupted.set()
            still_reader['value'] = lock.is_reader()
            lock.release_read_lock()

    # main thread keeps a second read lock so the upgrade has to wait
    lock.acquire_read_lock()
    thread = _daemon_thread(upgrader)
    thread.start()
    assert upgrader_ready.wait(WAIT_TIMEOUT)
    thread.join(WAIT_TIMEOUT)
    assert not thread.is_alive()
    assert interrupted.is_set()
    assert still_reader['value'] is True
    assert not lock.has_pending_writers

    lock.release_read_lock()
    # queue usable: a plain writer can now acquire
    lock.acquire_write_lock()
    assert lock.is_writer()
    lock.release_write_lock()
    assert lock.owner is None


def test_upgrade_with_custom_thread_identity():
    lock = fasteners.ReaderWriterLock(
        current_thread_functor=threading.get_ident)

    lock.acquire_read_lock()
    assert lock.upgrade_read_lock(timeout=0) is True
    assert lock.is_writer()
    lock.downgrade_write_lock()
    assert lock.is_reader()
    lock.release_read_lock()
    assert lock.owner is None


def test_plain_reader_to_writer_still_rejected_during_upgrades():
    lock = fasteners.ReaderWriterLock()

    lock.acquire_read_lock()
    with pytest.raises(RuntimeError):
        lock.acquire_write_lock()
    assert lock.is_reader()
    assert not lock.is_writer()
    lock.release_read_lock()
    assert lock.owner is None


def test_upgrade_downgrade_stress():
    lock = fasteners.ReaderWriterLock()
    watch = _utils.StopWatch(duration=2)
    watch.start()
    dups = collections.deque()
    state = {'active_writer': None, 'value': 0}

    def upgrading_reader(me):
        while not watch.expired():
            with lock.read_lock():
                if random.random() < 0.3 and lock.upgrade_read_lock(timeout=0):
                    try:
                        if state['active_writer'] is not None:
                            dups.append(me)
                        state['active_writer'] = me
                        state['value'] += 1
                        time.sleep(random.random() / 1000)
                        state['active_writer'] = None
                    finally:
                        lock.downgrade_write_lock()

    def plain_writer(me):
        while not watch.expired():
            lock.acquire_write_lock()
            if state['active_writer'] is not None:
                dups.append(me)
            state['active_writer'] = me
            state['value'] += 1
            time.sleep(random.random() / 1000)
            state['active_writer'] = None
            lock.release_write_lock()

    def plain_reader(unused_me):
        while not watch.expired():
            with lock.read_lock():
                # a genuine reader can never coexist with an active writer
                if state['active_writer'] is not None:
                    dups.append(unused_me)
                time.sleep(random.random() / 1000)

    threads = []
    for i in range(0, 6):
        threads.append(_daemon_thread(lambda i=i: upgrading_reader(i)))
    for i in range(0, 6):
        threads.append(_daemon_thread(lambda i=i: plain_reader(i)))
    for i in range(0, 4):
        threads.append(_daemon_thread(lambda i=i: plain_writer(i)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not dups
    assert state['active_writer'] is None
    assert state['value'] > 0
    assert lock.owner is None
    assert not lock.has_pending_writers
