"""Cooperative lock supervisor; no CUDA/model import and no public run CLI."""
import fcntl
import os
from pathlib import Path
import stat
import subprocess
import time


def supervise(lock_path, command, event, *, popen=subprocess.Popen):
    """Nonblocking lock; inherited fd remains locked until CUDA child exits.

    This supervises the direct child, not descendants that deliberately close
    the inherited descriptor. No replacement or deletion of lock file occurs.
    Journal failures before spawn prevent dispatch.
    """
    path=Path(lock_path)
    if (not path.is_absolute() or path.parent!=path.parent.resolve() or
            type(command) is not list or not command or
            any(type(x) is not str or not x for x in command) or not callable(event)):
        raise ValueError("canonical lock, fixed argv and durable callback required")
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:
            raise ValueError("regular single-link lock")
        begin=time.monotonic()
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=os.stat(path,follow_symlinks=False)
        if (st.st_dev,st.st_ino)!=(current.st_dev,current.st_ino):
            raise ValueError("cooperative lock inode replaced")
        event(dict(kind="lock_acquired",device=st.st_dev,inode=st.st_ino,
                   wait_seconds=time.monotonic()-begin,scope="cooperative_only"))
        event(dict(kind="child_spawn_intent",argv=command))
        try:
            child=popen(command,pass_fds=(fd,),close_fds=True)
        except BaseException as exc:
            event(dict(kind="child_spawn_failed",error_type=type(exc).__name__,message=str(exc)[:1000]))
            raise
        try:
            event(dict(kind="child_spawned",pid=child.pid))
            code=child.wait()
        except BaseException:
            # Do not unlock or terminate the child. Its inherited open-file
            # description continues to own the lock until it exits.
            raise
        event(dict(kind="child_exited",pid=child.pid,returncode=code))
        return code
    finally:
        # No LOCK_UN: the shared open-file description must survive in child.
        os.close(fd)
