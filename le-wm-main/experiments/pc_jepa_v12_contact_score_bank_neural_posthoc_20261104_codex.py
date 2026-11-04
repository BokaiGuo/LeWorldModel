"""Isolated predictor and actual-image scoring for the fixed contact bank."""
import json
import shutil
import subprocess

import numpy as np

from experiments import pc_jepa_v12_contact_score_bank_posthoc_20261104_codex as source

ROOT, OUT, SOURCE = source.ROOT, source.OUT, source.SOURCE
sha = source.feedback.sha
PREDICT_WORKER = ROOT / "experiments/pc_jepa_v12_contact_score_predict_worker_20261104_codex.py"
ENCODE_WORKER = ROOT / "experiments/pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex.py"


def launch(base, module):
    inp, out = base / "input", base / "output"
    inputs = [ROOT / "experiments", ROOT / ".venv", ROOT / "external/jepa-wms",
              ROOT / "external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8",
              ROOT / ".cache/dinov2/dinov2-small",
              ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar", inp]
    lock = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
    if any(not path.exists() for path in [*inputs, lock, out]):
        raise FileNotFoundError("missing isolated input")
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
                str(ROOT / ".venv/bin/python"), "-m", module, str(base)])
    subprocess.run(cmd, check=True, timeout=600)


def main():
    result_path, arrays_path = OUT / "result.json", OUT / "arrays.npz"
    result = json.loads(result_path.read_text())
    if result["status"] != "complete" or result["arrays_sha256"] != sha(arrays_path):
        raise ValueError("unverified physical bank")
    for path, digest in result["bindings"].items():
        if sha(path) != digest:
            raise ValueError("source dependency changed")
    base = OUT / "neural"
    base.mkdir()
    pred, enc = base / "predict", base / "encode"
    for sub in (pred, enc):
        sub.mkdir()
        (sub / "input").mkdir()
        (sub / "output").mkdir()
    source_input = SOURCE / "response_0/input"
    for name in ("context.npz", "config.json", "model_metadata.json"):
        shutil.copyfile(source_input / name, pred / "input" / name)
    shutil.copyfile(source_input / "model_metadata.json", enc / "input/model_metadata.json")
    with np.load(arrays_path, allow_pickle=False) as data:
        actions, images = data["actions"].copy(), data["terminal_rgb"].copy()
    with (pred / "input/bank.npy").open("xb") as handle:
        np.save(handle, actions)
    with (enc / "input/images.npz").open("xb") as handle:
        np.savez_compressed(handle, goal_rgb=np.load(SOURCE / "goal_rgb.npy", allow_pickle=False),
                            terminal_rgb=images)
    launch(pred, "experiments.pc_jepa_v12_contact_score_predict_worker_20261104_codex")
    launch(enc, "experiments.pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex")
    p = json.loads((pred / "output/decision.json").read_text())
    e = json.loads((enc / "output/encoded.json").read_text())
    if (p["worker_sha256"] != sha(PREDICT_WORKER) or p["bank_sha256"] != sha(pred / "input/bank.npy")
            or p["input_sha256"] != sha(source_input / "context.npz")
            or e["worker_sha256"] != sha(ENCODE_WORKER)
            or e["image_sha256"] != sha(enc / "input/images.npz")):
        raise ValueError("neural binding mismatch")
    predicted, encoded = np.asarray(p["decision"]["scores"]), np.asarray(e["scores"])
    if len(predicted) != 64 or len(encoded) != 64:
        raise ValueError("missing neural scores")
    original = np.asarray([r["model_score"] for r in result["rows"][:56]])
    groups = {}
    for group, sl in (("planner_slice", slice(0, 56)), ("witness_variants", slice(56, 64)),
                      ("all", slice(0, 64))):
        rows = result["rows"][sl]
        metrics = {key: np.asarray([row["metrics"][key] for row in rows])
                   for key in ("joint_xy_error", "object_error", "angle_error_radians")}
        groups[group] = {"predicted_vs_encoded": source.order(predicted[sl], encoded[sl]),
            "predicted_vs_physical": {k: source.order(predicted[sl], v) for k, v in metrics.items()},
            "encoded_vs_physical": {k: source.order(encoded[sl], v) for k, v in metrics.items()},
            "successful": sum(r["metrics"]["success"] for r in rows)}
    report = {"status": "complete", "source_result_sha256": sha(result_path),
        "predict_sha256": sha(pred / "output/decision.json"), "encoder_sha256": sha(enc / "output/encoded.json"),
        "max_abs_original_rescore_difference": float(np.max(np.abs(original - predicted[:56]))),
        "predicted": predicted.tolist(), "actual_encoded": encoded.tolist(), "groups": groups,
        "scope": "one exposed goal, correlated actions; privileged variants not planner candidates"}
    with (base / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
