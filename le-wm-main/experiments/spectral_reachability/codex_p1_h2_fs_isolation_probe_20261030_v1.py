#!/usr/bin/env python3
"""CPU-only bubblewrap filesystem boundary probe; never launches H2 or CUDA.

This is an acceptance diagnostic, not a production launcher or a security
boundary against a malicious kernel, privileged host user, or mutable mounts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


CHILD = r'''
import json, os, stat, sys
checks = {}
def denied(path):
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError:
        return True
    return False
checks["allowed_read"] = open("/inputs/allowed", "rb").read() == b"allowed"
checks["hidden_absolute"] = denied(sys.argv[1])
checks["hidden_symlink"] = denied("/inputs/escape")
checks["hidden_proc_root"] = denied("/proc/1/root" + sys.argv[1])
checks["host_home_hidden"] = denied("/home/gbk/.ssh/id_rsa")
try:
    open("/inputs/allowed", "wb").close()
    checks["readonly"] = False
except OSError:
    checks["readonly"] = True
with open("/output/child.txt", "x") as f:
    f.write("isolated")
checks["output_write"] = True
checks["no_inherited_extra_fds"] = all(
    not os.path.exists("/proc/self/fd/" + str(i)) for i in range(3, 64))
checks["device_nodes"] = all(stat.S_ISCHR(os.stat(p).st_mode) for p in sys.argv[2:])
print(json.dumps(checks, sort_keys=True))
sys.exit(0 if all(checks.values()) else 1)
'''


def run_probe(report, devices=()):
    report = Path(report)
    # Reserve the report before running; never overwrite prior evidence.
    with report.open("x", encoding="utf-8") as out:
        evidence = {"schema": "h2-fs-isolation-probe-v1", "status": "failed",
                    "production_authorized": False, "cuda_initialized": False,
                    "network_isolated": False, "devices": list(devices)}
        try:
            for device in devices:
                p = Path(device)
                if not p.is_absolute() or not p.name.startswith("nvidia") or p.parent != Path("/dev"):
                    raise ValueError("devices must be explicit /dev/nvidia* paths")
                if not stat.S_ISCHR(p.stat().st_mode) or p.is_symlink():
                    raise ValueError("device must be a non-symlink character device")
            with tempfile.TemporaryDirectory(prefix="h2-fs-probe-") as tmp:
                root = Path(tmp)
                (root / "allowed").write_bytes(b"allowed")
                (root / "hidden").write_bytes(b"hidden")
                (root / "output").mkdir()
                # This target is deliberately absent from the child's mount namespace.
                os.symlink(str(root / "hidden"), root / "escape")
                cmd = ["/usr/bin/bwrap", "--unshare-user", "--unshare-pid",
                       "--die-with-parent", "--new-session",
                       "--ro-bind", "/usr", "/usr",
                       "--ro-bind", "/lib", "/lib",
                       "--ro-bind", "/lib64", "/lib64",
                       "--proc", "/proc", "--dev", "/dev",
                       "--dir", "/inputs", "--dir", "/output",
                       "--ro-bind", str(root / "allowed"), "/inputs/allowed",
                       "--symlink", str(root / "hidden"), "/inputs/escape",
                       "--bind", str(root / "output"), "/output"]
                for device in devices:
                    cmd += ["--dev-bind", device, device]
                cmd += ["--chdir", "/", "--", "/usr/bin/python3", "-I", "-c",
                        CHILD, str(root / "hidden"), *devices]
                result = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, timeout=30, close_fds=True,
                                        env={"PATH": "/usr/bin:/bin", "LANG": "C"})
                evidence["exit_code"] = result.returncode
                evidence["stderr"] = result.stderr[-4000:]
                if result.returncode == 0:
                    evidence["checks"] = json.loads(result.stdout)
                    evidence["output_sha256"] = hashlib.sha256(
                        (root / "output" / "child.txt").read_bytes()).hexdigest()
                    evidence["status"] = "passed"
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            evidence["error"] = str(exc)
        json.dump(evidence, out, sort_keys=True, indent=2)
        out.write("\n")
    return evidence["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="new audit JSON, never overwritten")
    parser.add_argument("--device", action="append", default=[],
                        help="optional explicit NVIDIA character device; stat only, no open")
    args = parser.parse_args()
    try:
        return 0 if run_probe(args.report, args.device) else 1
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
