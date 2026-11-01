#!/usr/bin/env python3
"""Synthetic Landlock feasibility probe, not a production policy or launcher."""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

LIBC = ctypes.CDLL(None, use_errno=True)
CREATE, ADD, RESTRICT = 444, 445, 446
VERSION = 1
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = (1 << i for i in range(4))
# ABI 1 filesystem rights. ABI 3 adds TRUNCATE, ABI 2 adds REFER.
ABI1 = (1 << 13) - 1
REFER, TRUNCATE = 1 << 13, 1 << 14


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


def call(number, *args):
    result = LIBC.syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def abi_version():
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("unverified syscall numbering on this architecture")
    return call(CREATE, None, 0, VERSION)


def apply(allowed_file):
    abi = abi_version()
    if abi < 3:
        raise RuntimeError("Landlock ABI >= 3 required for truncate protection")
    handled = ABI1 | REFER | TRUNCATE
    attr = Ruleset(handled)
    ruleset = call(CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    try:
        fd = os.open(allowed_file, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = PathRule(READ_FILE, fd)
            call(ADD, ruleset, 1, ctypes.byref(rule), 0)
        finally:
            os.close(fd)
        if LIBC.prctl(38, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        call(RESTRICT, ruleset, 0)
    finally:
        os.close(ruleset)
    return abi


def child(allowed, hidden):
    abi = apply(allowed)
    checks = {"allowed_read": Path(allowed).read_bytes() == b"allowed"}
    for name, path, flags in (
        ("hidden_read_denied", hidden, os.O_RDONLY),
        ("allowed_write_denied", allowed, os.O_WRONLY),
        ("allowed_truncate_denied", allowed, os.O_WRONLY | os.O_TRUNC),
    ):
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            checks[name] = exc.errno in (errno.EACCES, errno.EPERM)
        else:
            os.close(fd)
            checks[name] = False
    print(json.dumps({"abi": abi, "checks": checks}))
    return 0 if all(checks.values()) else 1


def probe():
    with tempfile.TemporaryDirectory(prefix="h2-landlock-probe-") as directory:
        allowed = Path(directory) / "allowed"
        hidden = Path(directory) / "hidden"
        allowed.write_bytes(b"allowed")
        hidden.write_bytes(b"hidden")
        return subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()),
             "--child", str(allowed), str(hidden)],
            env={"PATH": "/usr/bin:/bin", "LANG": "C"},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, close_fds=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(*args.child)
    result = probe()
    print(json.dumps({"returncode": result.returncode, "stdout": result.stdout,
                      "stderr": result.stderr, "production_authorized": False,
                      "cuda_tested": False}, indent=2))
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
