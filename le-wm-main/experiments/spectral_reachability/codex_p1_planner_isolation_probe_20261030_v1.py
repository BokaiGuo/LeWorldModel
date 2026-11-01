"""Minimal bwrap filesystem isolation probe, not a CUDA planner launcher.

No repository, checkpoint, GPU device or host /proc is mounted. Networking is
not isolated. Only trusted, explicitly mounted code may be run by this helper.
"""
import os
from pathlib import Path
import subprocess


def probe(script, payload, *, timeout=10):
    path = Path(script)
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or not path.is_file() or not isinstance(payload, bytes)
            or len(payload) > 2 * 1024 * 1024 or not 0 < timeout <= 30):
        raise ValueError("invalid probe inputs")
    argv = ["/usr/bin/bwrap", "--unshare-user", "--unshare-pid", "--die-with-parent",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
            "--dir", "/work", "--ro-bind", str(path), "/work/probe.py",
            "--chdir", "/work", "/usr/bin/python3", "-I", "/work/probe.py"]
    return subprocess.run(argv, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=False, close_fds=True,
                          env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
