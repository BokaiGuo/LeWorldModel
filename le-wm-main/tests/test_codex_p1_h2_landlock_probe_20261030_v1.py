import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_p1_h2_landlock_probe_20261030_v1.py"
spec = importlib.util.spec_from_file_location("landlock_probe", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_kernel_enforces_synthetic_read_and_write_policy():
    result = module.probe()
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["abi"] >= 3
    assert all(evidence["checks"].values())


def test_syscall_error_is_not_success():
    try:
        module.call(-1)
    except OSError:
        return
    assert False, "invalid syscall unexpectedly succeeded"
