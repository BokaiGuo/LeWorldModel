import fcntl
import os
import subprocess
import sys

import pytest
from experiments.spectral_reachability import codex_p2_program_library_supervisor_20261030_v1 as s


def test_lock_held_during_child_and_release_after(tmp_path):
    lock=tmp_path/"gpu.lock";lock.touch();events=[]
    def spawn(argv,**kw):
        fd=os.open(lock,os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert len(kw["pass_fds"])==1
        finally:
            os.close(fd)
        return subprocess.Popen(argv,**kw)
    assert s.supervise(lock,[sys.executable,"-c","raise SystemExit(7)"],events.append,popen=spawn)==7
    assert [e["kind"] for e in events]==["lock_acquired","child_spawn_intent","child_spawned","child_exited"]
    fd=os.open(lock,os.O_RDWR)
    try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally: os.close(fd)


def test_busy_lock_never_spawns(tmp_path):
    lock=tmp_path/"gpu.lock";lock.touch();fd=os.open(lock,os.O_RDWR)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            s.supervise(lock,[sys.executable,"-c","pass"],lambda _:None,
                        popen=lambda *a,**kw:pytest.fail("spawned while busy"))
    finally: os.close(fd)


def test_journal_failure_before_spawn(tmp_path):
    lock=tmp_path/"gpu.lock";lock.touch()
    def fail(_): raise OSError("journal failed")
    with pytest.raises(OSError):
        s.supervise(lock,[sys.executable,"-c","pass"],fail,
                    popen=lambda *a,**kw:pytest.fail("spawn after journal failure"))
