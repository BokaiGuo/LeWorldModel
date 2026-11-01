"""Blocking cooperative advisory lock; waiting is not a completed run."""
import contextlib
import fcntl
import os
from pathlib import Path
import stat
import time


@contextlib.contextmanager
def gpu_lock(path):
    path = Path(path)
    if not path.is_absolute() or path.parent != path.parent.resolve():
        raise ValueError("canonical absolute lock parent required")
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    locked = False
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("single-link regular lock required")
        start = time.monotonic_ns()
        fcntl.flock(fd, fcntl.LOCK_EX)
        waited = (time.monotonic_ns() - start) / 1e9
        locked = True
        current = os.stat(path, follow_symlinks=False)
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("lock file replaced while waiting")
        yield waited
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
