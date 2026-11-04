"""Prepare and isolate frozen-encoder scoring of actual terminal RGBs."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from experiments.pc_jepa_v11_same_bank_physical_score_diag_20261104_codex import OUT, ROOT, SOURCE, order
from experiments.pc_jepa_native_v11_feasible_feedback_dev_20261104_codex import sha


def main():
    result = json.loads((OUT / "result.json").read_text())
    arrays_path = OUT / "arrays.npz"
    if result["status"] != "complete" or result["arrays_sha256"] != sha(arrays_path):
        raise ValueError("unverified physical bank")
    base = OUT / "encoder"
    base.mkdir()
    inp, out = base / "input", base / "output"
    inp.mkdir()
    out.mkdir()
    with np.load(arrays_path, allow_pickle=False) as package:
        frames = package["terminal_rgb"].copy()
    goal_path = SOURCE / "goal_rgb.npy"
    with (inp / "images.npz").open("xb") as handle:
        np.savez_compressed(handle, goal_rgb=np.load(goal_path, allow_pickle=False), terminal_rgb=frames)
    metadata = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex/input/model_metadata.json"
    with (inp / "model_metadata.json").open("x") as handle:
        json.dump(json.loads(metadata.read_text()), handle, allow_nan=False)
    worker = ROOT / "experiments/pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex.py"
    inputs = [ROOT / "experiments", ROOT / ".venv", ROOT / "external/jepa-wms",
              ROOT / "external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
              ROOT / ".cache/dinov2/dinov2-small", ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar", inp]
    lock = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
    if any(not path.exists() for path in [*inputs, lock, out]):
        raise FileNotFoundError("missing isolated encoder input")
    cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net", "--ro-bind", "/", "/",
           "--tmpfs", "/home/gbk", "--tmpfs", "/tmp", "--proc", "/proc", "--dev-bind", "/dev", "/dev"]
    dirs = set()
    for path in [*inputs, lock, out]:
        parent = path.parent
        while str(parent).startswith("/home/gbk/"):
            dirs.add(parent)
            parent = parent.parent
    for path in sorted(dirs, key=lambda p: (len(p.parts), str(p))):
        cmd.extend(["--dir", str(path)])
    for path in inputs:
        cmd.extend(["--ro-bind", str(path), str(path)])
    for path in (lock, out):
        cmd.extend(["--bind", str(path), str(path)])
    cmd.extend(["--chdir", str(ROOT), "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "HOME", "/tmp", "--setenv", "PYTHONPATH", str(ROOT),
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "CUDA_VISIBLE_DEVICES", "0",
                str(ROOT / ".venv/bin/python"), "-m",
                "experiments.pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex", str(base)])
    subprocess.run(cmd, check=True, timeout=600)
    encoded = json.loads((out / "encoded.json").read_text())
    if (encoded["image_sha256"] != sha(inp / "images.npz")
            or encoded["metadata_sha256"] != sha(inp / "model_metadata.json")
            or encoded["worker_sha256"] != sha(worker) or len(encoded["scores"]) != 64):
        raise ValueError("encoder binding mismatch")
    actual = np.asarray(encoded["scores"])
    model = np.asarray([row["model_score"] for row in result["rows"]])
    physical = {key: np.asarray([row["metrics"][key] for row in result["rows"]])
                for key in ("joint_xy_error", "object_error", "angle_error_radians")}
    report = {"status": "complete", "source_result_sha256": sha(OUT / "result.json"),
              "source_arrays_sha256": sha(arrays_path), "goal_sha256": sha(goal_path),
              "encoded_sha256": sha(out / "encoded.json"), "worker_sha256": sha(worker),
              "model_vs_actual_encoded": order(model, actual),
              "actual_encoded_vs_physical": {key: order(actual, value) for key, value in physical.items()},
              "scope": "one correlated bank and constructed goal; no PC gate claim"}
    with (base / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
