"""Advisory single-host GPU lock; no model import or production entry point."""
import contextlib
import fcntl
import os
import stat
from pathlib import Path


@contextlib.contextmanager
def gpu_lock(path):
    """Hold an existing stable regular lock file for the entire GPU operation.

    All participants must agree on the same absolute path and never unlink or
    replace the file. This is advisory, not a device-wide reservation.
    """
    path = Path(path)
    if not path.is_absolute() or path.parent != path.parent.resolve():
        raise ValueError("canonical absolute lock parent required")
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    locked = False
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise ValueError("lock must be a regular single-link file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        current = os.stat(path, follow_symlinks=False)
        if (st.st_dev, st.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("lock file replaced")
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
