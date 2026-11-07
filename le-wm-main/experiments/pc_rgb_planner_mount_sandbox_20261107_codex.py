"""Run the sealed RGB planner in an OS mount namespace with restricted project files."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments import pc_rgb_gradient_public_plan_20261107_codex as prior
from experiments.pc_input_permissions import file_hash

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_PLANNER_MOUNT_SANDBOX_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_planner_mount_sandbox_20261107_codex.json"
SEALED = ROOT / "results/pc_rgb_planner_mount_sandbox_plan_20261107_codex.json"

WORKER = r"""
import json, os
from pathlib import Path
paths = json.loads(os.environ["PC_DENIED_PATHS"])
checks = {p: os.access(p, os.R_OK) for p in paths}
if any(checks.values()):
    raise RuntimeError("protected path was readable")
from experiments import pc_rgb_gradient_public_plan_20261107_codex as planner
planner.OUT = Path("/output/plan.json")
planner.main()
Path("/output/probes.json").write_text(json.dumps(checks))
"""


def make_command(output_dir):
    root = str(ROOT)
    home = "/home/gbk"
    denied = [
        "/home/gbk/.stable_worldmodel/datasets/tworoom.h5",
        str(ROOT / "results/pc_rgb_gradient_public_evaluation_20261107_codex.json"),
        str(ROOT / "results/pc_rgb_gradient_public_plan_20261107_codex.json"),
        str(ROOT / "results/pc_tworoom_goal_gradient_development_physical_20261107_codex.json"),
    ]
    mounts = [
        ("--ro-bind", "/usr", "/usr"), ("--ro-bind", "/lib", "/lib"),
        ("--ro-bind", "/lib64", "/lib64"), ("--ro-bind", "/etc", "/etc"),
        ("--ro-bind", "/sys", "/sys"), ("--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia"),
        ("--ro-bind", root + "/.venv", root + "/.venv"),
        ("--ro-bind", root + "/experiments", root + "/experiments"),
        ("--ro-bind", root + "/eval.py", root + "/eval.py"),
        ("--ro-bind", str(prior.PUBLIC), str(prior.PUBLIC)),
        ("--ro-bind", str(prior.PROTOCOL), str(prior.PROTOCOL)),
        ("--ro-bind", str(prior.CHECKPOINT), str(prior.CHECKPOINT)),
        ("--bind", str(output_dir), "/output"),
    ]
    command = ["bwrap", "--die-with-parent", "--unshare-pid", "--proc", "/proc",
               "--dev-bind", "/dev", "/dev", "--tmpfs", "/tmp"]
    for option, src, dst in mounts:
        if not Path(src).exists():
            raise FileNotFoundError(src)
        command.extend((option, src, dst))
    command.extend(("--chdir", root, "--setenv", "HOME", "/tmp",
                    "--setenv", "XDG_CACHE_HOME", "/tmp/cache",
                    "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                    "--setenv", "PYTHONPATH", root,
                    "--setenv", "PC_DENIED_PATHS", json.dumps(denied),
                    root + "/.venv/bin/python", "-c", WORKER))
    return command, denied, mounts


def compare(new, reference):
    if new["public_sha256"] != reference["public_sha256"] or new["model_digest"] != reference["model_digest"]:
        raise ValueError("model or public input differs")
    rows = []
    for a, b in zip(new["rows"], reference["rows"], strict=True):
        if a["episode"] != b["episode"]:
            raise ValueError("episode mismatch")
        errors = {}
        for name in ("candidates", "direction", "gradient", "scores"):
            x, y = np.asarray(a[name]), np.asarray(b[name])
            if x.shape != y.shape or not np.allclose(x, y, rtol=1e-5, atol=1e-3):
                raise ValueError(f"{name} differs")
            errors[name] = float(np.max(np.abs(x-y)))
        if a["selected_index"] != b["selected_index"]:
            raise ValueError("selection differs")
        rows.append({"episode": a["episode"], "selected_index": a["selected_index"], "max_abs_error": errors})
    return rows


def main():
    if OUT.exists() or SEALED.exists():
        raise FileExistsError("sandbox result already exists")
    reference = json.loads(prior.OUT.read_text())
    if reference["public_sha256"] != file_hash(prior.PUBLIC) or reference["runner_sha256"] != file_hash(prior.__file__):
        raise ValueError("reference source drift")
    tmp = Path(tempfile.mkdtemp(prefix="pc-rgb-sandbox-"))
    command, denied, mounts = make_command(tmp)
    print("Run frozen RGB planner with project data and evaluation results absent from mount namespace.", flush=True)
    result = subprocess.run(command, check=False, close_fds=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:], flush=True)
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"sandbox exited {result.returncode}; temporary artifacts: {tmp}")
    probe = json.loads((tmp / "probes.json").read_text())
    if set(probe) != set(denied) or any(probe.values()):
        raise ValueError("protected path probe failure")
    new = json.loads((tmp / "plan.json").read_text())
    rows = compare(new, reference)
    with SEALED.open("x") as stream:
        json.dump(new, stream, separators=(",", ":"), allow_nan=False)
    with OUT.open("x") as stream:
        json.dump({"scope": "whitelist mount test, not network or side-channel isolation",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "sealed_plan_sha256": file_hash(SEALED), "reference_sha256": file_hash(prior.OUT),
                   "public_sha256": file_hash(prior.PUBLIC), "denied_paths_readable": probe,
                   "mounts": mounts, "unshared_pid": True, "network_namespace": False,
                   "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-3000:],
                   "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"sealed {len(rows)} sandboxed RGB plans; protected path probes denied", flush=True)


if __name__ == "__main__":
    main()
