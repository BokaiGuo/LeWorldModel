"""CPU-only synthetic evidence: child inheritance and pre-opened FD hazard."""
from pathlib import Path
import subprocess
import sys


PROBE = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_p1_h2_landlock_probe_20261030_v1.py"

PROGRAM = r"""
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
allowed, hidden = sys.argv[2:]
fd = os.open(hidden, os.O_RDONLY)
probe.apply(allowed)
try:
    os.open(hidden, os.O_RDONLY)
except PermissionError:
    pass
else:
    raise AssertionError("new open must be denied")
assert os.read(fd, 6) == b"hidden", "pre-opened FD remains readable"
os.close(fd)
pid = os.fork()
if pid == 0:
    try:
        os.open(hidden, os.O_RDONLY)
    except PermissionError:
        os._exit(0)
    os._exit(1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, "fork child escaped policy"
"""


def test_fork_inherits_policy_but_preopened_fd_bypasses_it(tmp_path):
    allowed, hidden = tmp_path / "allowed", tmp_path / "hidden"
    allowed.write_bytes(b"allowed")
    hidden.write_bytes(b"hidden")
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROGRAM, str(PROBE), str(allowed), str(hidden)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, close_fds=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C"}, timeout=15,
    )
    assert result.returncode == 0, result.stderr
