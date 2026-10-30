# -*- coding: utf-8 -*-

# Copyright (C) 2014 Yahoo! Inc. All Rights Reserved.
# Copyright 2011 OpenStack Foundation.
#
# All Rights Reserved.
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
import contextlib
import functools
import threading
import time
from typing import Optional

from fasteners import _utils


class _UpgradeRequest(object):
    """Marker queued by a reader waiting to become the writer.

    A plain entry in ``ReaderWriterLock._pending_writers`` is the bare
    identity of a thread waiting for a write lock. Entries of this type
    reserve the queue position of a reader that is atomically upgrading the
    single read lock it already holds into a write lock.
    """

    __slots__ = ('owner',)

    def __init__(self, owner):
        self.owner = owner


class ReaderWriterLock(object):
    """An inter-thread readers writer lock."""

    WRITER = 'w'  #: Writer owner type/string constant.
    READER = 'r'  #: Reader owner type/string constant.

    def __init__(self,
                 condition_cls=threading.Condition,
                 current_thread_functor=threading.current_thread):
        """
        Args:
            condition_cls:
                Optional custom `Condition` primitive used for synchronization.
            current_thread_functor:
                Optional function that returns the identity of the thread in case
                threads are not properly identified by threading.current_thread
        """
        self._writer = None
        self._writer_entries = 0
        self._pending_writers = collections.deque()
        self._readers = {}
        self._cond = condition_cls()
        self._current_thread = current_thread_functor

    def _pending_index(self, me):
        """Return the queue index of ``me``'s pending write/upgrade entry.

        Returns ``None`` when ``me`` is not queued.
        """
        for index, entry in enumerate(self._pending_writers):
            if isinstance(entry, _UpgradeRequest):
                owner = entry.owner
            else:
                owner = entry
            if owner == me:
                return index
        return None

    @property
    def has_pending_writers(self) -> bool:
        """Check if there pending writers

        Returns:
            Whether there are pending writers.
        """
        return bool(self._pending_writers)

    def is_writer(self, check_pending: bool = True) -> bool:
        """Check if caller is a writer (optionally pending writer).

        A reader that is waiting for its :meth:`upgrade_read_lock` request to
        be granted is also considered a pending writer.

        Args:
            check_pending:
                Whether to check for pending writer status.

        Returns:
            Whether the caller is the active (or optionally pending) writer.
        """
        me = self._current_thread()
        if self._writer == me:
            return True
        if check_pending:
            return self._pending_index(me) is not None
        else:
            return False

    def is_reader(self) -> bool:
        """Check if caller is a reader.

        Returns:
            Whether the caller is an active reader.
        """
        me = self._current_thread()
        return me in self._readers

    @property
    def owner(self) -> Optional[str]:
        """Caller ownership (if any) of the lock

        Returns:
            `'w'` if caller is a writer, `'r'` if caller is a reader, None otherwise.
        """
        """Returns whether the lock is locked by a writer or reader."""
        if self._writer is not None:
            return self.WRITER
        if self._readers:
            return self.READER
        return None

    def acquire_read_lock(self):
        """Acquire a read lock.

        Will wait until no active or pending writers.

        Raises:
            RuntimeError: if a pending writer tries to acquire a read lock.
        """
        me = self._current_thread()
        self._acquire_read_lock(me)

    def release_read_lock(self):
        """Release a read lock.

        Raises:
            RuntimeError: if the current thread does not own a read lock.
        """
        me = self._current_thread()
        self._release_read_lock(me)

    def _acquire_read_lock(self, me):
        if self._pending_index(me) is not None:
            raise RuntimeError("Writer %s can not acquire a read lock"
                               " while waiting for the write lock"
                               % me)
        with self._cond:
            while True:
                # No active writer, or we are the writer;
                # Also no pending writers;
                # we are good to become a reader.
                if self._writer is None or self._writer == me:
                    if me in self._readers:
                        # ok to get a lock if current thread already has one
                        self._readers[me] = self._readers[me] + 1
                        break
                    elif (self._writer == me) or not self.has_pending_writers:
                        self._readers[me] = 1
                        break
                # An active or pending writer; guess we have to wait.
                self._cond.wait()

    def _release_read_lock(self, me, raise_on_not_owned=True):
        # I am no longer a reader, remove *one* occurrence of myself.
        # If the current thread acquired two read locks, then it will
        # still have to remove that other read lock; this allows for
        # basic reentrancy to be possible.
        with self._cond:
            try:
                me_instances = self._readers[me]
                if me_instances > 1:
                    self._readers[me] = me_instances - 1
                else:
                    self._readers.pop(me)
            except KeyError:
                if raise_on_not_owned:
                    raise RuntimeError(f"Thread {me} does not own a read lock")
            self._cond.notify_all()

    @contextlib.contextmanager
    def read_lock(self):
        """Context manager that grants a read lock.

        Will wait until no active or pending writers.

        Raises:
            RuntimeError: if a pending writer tries to acquire a read lock.
        """
        me = self._current_thread()
        self._acquire_read_lock(me)
        try:
            yield self
        finally:
            self._release_read_lock(me, raise_on_not_owned=False)

    def _acquire_write_lock(self, me):
        if self.is_reader():
            raise RuntimeError("Reader %s to writer privilege"
                               " escalation not allowed" % me)

        with self._cond:
            self._pending_writers.append(me)
            while True:
                # No readers, and no active writer, am I next??
                if len(self._readers) == 0 and self._writer is None:
                    head = self._pending_writers[0]
                    # An upgrade request is only granted through
                    # upgrade_read_lock(); a plain writer never takes its
                    # slot, and a request's owner stays a reader until the
                    # conversion happens, so it can not reach this branch
                    # with such an entry at the head.
                    if not isinstance(head, _UpgradeRequest) and head == me:
                        self._writer = self._pending_writers.popleft()
                        self._writer_entries = 1
                        break
                self._cond.wait()

    def _release_write_lock(self, me, raise_on_not_owned=True):
        with self._cond:
            self._writer = None
            self._writer_entries = 0
            self._cond.notify_all()

    def acquire_write_lock(self):
        """Acquire a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Guaranteed for locks to be processed in fair order (FIFO).

        Raises:
            RuntimeError: if an active reader attempts to acquire a lock.
        """
        me = self._current_thread()
        if self._writer == me:
            self._writer_entries += 1
        else:
            self._acquire_write_lock(me)

    def release_write_lock(self):
        """Release a write lock.

        Raises:
            RuntimeError: if the current thread does not own a write lock.
        """
        me = self._current_thread()
        if self._writer == me:
            self._writer_entries -= 1
            if self._writer_entries == 0:
                self._release_write_lock(me)
        else:
            raise RuntimeError(f"Thread {me} does not own a write lock")

    def _can_grant_upgrade(self, me, request):
        """Whether ``request`` may be converted into a write lock now.

        The caller keeps holding its read lock while waiting, so it is the
        only reader that may remain; every reader that entered before the
        upgrade request must have released first.
        """
        return (self._writer is None
                and len(self._pending_writers) > 0
                and self._pending_writers[0] is request
                and len(self._readers) == 1
                and me in self._readers
                and self._readers[me] == 1)

    def _grant_upgrade(self, me, request):
        """Atomically turn ``me``'s read lock into a write lock.

        Must be called while holding ``self._cond`` with
        :meth:`_can_grant_upgrade` satisfied.
        """
        self._pending_writers.popleft()
        del self._readers[me]
        self._writer = me
        self._writer_entries = 1
        self._cond.notify_all()

    def upgrade_read_lock(self, timeout=None) -> bool:
        """Atomically upgrade the single read lock held by the caller.

        The caller must own exactly one read lock (and no write lock). On
        success that read lock is converted into a write lock, held exactly
        once, without ever leaving the lock unprotected; use
        :meth:`downgrade_write_lock` to convert it back.

        While the upgrade is pending, readers that entered before the
        request may finish normally, but no *new* readers are admitted and
        no queued writer loses its place: the upgrade request is queued
        after every writer that is already waiting.

        Args:
            timeout:
                ``None`` (the default) waits until the upgrade can be
                granted. A non-negative number bounds the wait to that many
                seconds; ``0`` only performs an upgrade that can be granted
                immediately.

        Returns:
            ``True`` if the read lock was converted into a write lock,
            ``False`` if the upgrade could not be granted: ``timeout``
            expired (or was ``0`` and the upgrade could not happen at once),
            a plain writer was already queued ahead, or another reader was
            already waiting to upgrade. On ``False`` the caller keeps
            holding its read lock and the lock queue is left untouched.

        Raises:
            RuntimeError: if the caller does not hold exactly one read lock
                (it holds none, nested read locks, or the write lock).
            ValueError: if ``timeout`` is negative.
        """
        me = self._current_thread()
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        with self._cond:
            if self._writer == me:
                raise RuntimeError("Writer %s can not upgrade a read lock"
                                   " it does not hold as a plain reader"
                                   % me)
            read_entries = self._readers.get(me, 0)
            if read_entries == 0:
                raise RuntimeError("Thread %s does not own a read lock"
                                   % me)
            if read_entries != 1:
                raise RuntimeError("Reader %s may only upgrade while holding"
                                   " exactly one read lock" % me)

            # Only one reader may wait for an upgrade at a time, and an
            # upgrade never overtakes an already queued plain writer.
            if any(isinstance(entry, _UpgradeRequest)
                   for entry in self._pending_writers):
                return False
            if self._pending_writers:
                return False

            # Fast path: the queue is empty and no other reader is active,
            # so the single read lock becomes a write lock right away.
            if self._writer is None and len(self._readers) == 1:
                del self._readers[me]
                self._writer = me
                self._writer_entries = 1
                self._cond.notify_all()
                return True

            if timeout == 0:
                return False

            request = _UpgradeRequest(me)
            self._pending_writers.append(request)
            granted = False
            try:
                if timeout is None:
                    while not self._can_grant_upgrade(me, request):
                        self._cond.wait()
                else:
                    deadline = time.monotonic() + timeout
                    while not self._can_grant_upgrade(me, request):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return False
                        self._cond.wait(remaining)
                self._grant_upgrade(me, request)
                granted = True
                return True
            finally:
                # On timeout or an interrupted wait the upgrade intent must
                # disappear: the caller keeps its read lock, and any writer
                # queued behind stays able to proceed once readers drain.
                if not granted:
                    try:
                        self._pending_writers.remove(request)
                    except ValueError:
                        pass
                    self._cond.notify_all()

    def downgrade_write_lock(self):
        """Atomically downgrade the single write lock held by the caller.

        The caller must own exactly one write lock (and no read lock). The
        write lock is converted into a read lock in one atomic step, so a
        writer queued behind the caller can not slip in between.

        Raises:
            RuntimeError: if the caller does not hold exactly one write lock
                (it holds none, nested write locks, or a read lock as well);
                the lock state is left unchanged.
        """
        me = self._current_thread()
        with self._cond:
            if self._writer != me:
                raise RuntimeError(f"Thread {me} does not own a write lock")
            if self._writer_entries != 1:
                raise RuntimeError("Writer %s may only downgrade while"
                                   " holding exactly one write lock" % me)
            if me in self._readers:
                raise RuntimeError("Writer %s may not downgrade while also"
                                   " holding a read lock" % me)
            self._writer = None
            self._writer_entries = 0
            self._readers[me] = 1
            self._cond.notify_all()

    @contextlib.contextmanager
    def write_lock(self):
        """Context manager that grants a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Guaranteed for locks to be processed in fair order (FIFO).

        Raises:
            RuntimeError: if an active reader attempts to acquire a lock.
        """
        me = self._current_thread()
        if self.is_writer(check_pending=False):
            self._writer_entries += 1
            try:
                yield self
            finally:
                self._writer_entries -= 1
        else:
            self._acquire_write_lock(me)
            try:
                yield self
            finally:
                self._release_write_lock(me)


def locked(*args, **kwargs):
    """A locking **method** decorator.

    It will look for a provided attribute (typically a lock or a list
    of locks) on the first argument of the function decorated (typically this
    is the 'self' object) and before executing the decorated function it
    activates the given lock or list of locks as a context manager,
    automatically releasing that lock on exit.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    the lock/list of locks object/s) in the instance object this decorator
    is attached to.

    NOTE(harlowja): a custom logger (which will be used if lock release
    failures happen) can be provided by passing a logger instance for keyword
    argument ``logger``.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')
        logger = kwargs.get('logger')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, (tuple, list)):
                with _utils.LockStack(logger=logger) as stack:
                    for i, lock in enumerate(attr_value):
                        if not stack.acquire_lock(lock):
                            raise threading.ThreadError("Unable to acquire"
                                                        " lock %s" % (i + 1))
                    return f(self, *args, **kwargs)
            else:
                lock = attr_value
                with lock:
                    return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


def read_locked(*args, **kwargs):
    """Acquires & releases a read lock around call into decorated method.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    a :py:class:`.ReaderWriterLock`) in the instance object this decorator
    is attached to.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            rw_lock = getattr(self, attr_name)
            with rw_lock.read_lock():
                return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


def write_locked(*args, **kwargs):
    """Acquires & releases a write lock around call into decorated method.

    NOTE(harlowja): if no attribute name is provided then by default the
    attribute named '_lock' is looked for (this attribute is expected to be
    a :py:class:`.ReaderWriterLock` object) in the instance object this
    decorator is attached to.

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """

    def decorator(f):
        attr_name = kwargs.get('lock', '_lock')

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            rw_lock = getattr(self, attr_name)
            with rw_lock.write_lock():
                return f(self, *args, **kwargs)

        return wrapper

    # This is needed to handle when the decorator has args or the decorator
    # doesn't have args, python is rather weird here...
    if kwargs or not args:
        return decorator
    else:
        if len(args) == 1:
            return decorator(args[0])
        else:
            return decorator


@contextlib.contextmanager
def try_lock(lock: threading.Lock) -> bool:
    """Context manager that attempts to acquire a lock without a timeout, and
    releases it on exit (if acquired).

    Args:
        lock:
            A lock to try to acquire.

    Returns:
        Whether the lock was acquired.

    # NOTE(harlowja): the keyword argument for 'blocking' does not work
    # in py2.x and only is fixed in py3.x (this adjustment is documented
    # and/or debated in http://bugs.python.org/issue10789); so we'll just
    # stick to the format that works in both (oddly the keyword argument
    # works in py2.x but only with reentrant locks).

    NOTE(paulius): This function is DEPRECATED and will be kept until the end
    of time. It is potentially used by oslo, but too specific to be recommended
    for other projects
    """
    was_locked = lock.acquire(False)
    try:
        yield was_locked
    finally:
        if was_locked:
            lock.release()
