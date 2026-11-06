"""Run and summarize the sealed contact-bank latent error diagnostic."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from experiments import pc_jepa_v12_contact_score_bank_neural_posthoc_20261104_codex as neural
from experiments.audit_pc_jepa_v22_unqueried_suffix_dev_20261105_codex import independent_order
from experiments.pc_jepa_v22_contact_feedback_dev_20261104_codex import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_v22_privileged_contact_suffix_20261105_codex"
OUT = ROOT / "results/pc_jepa_v22_contact_latent_error_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_CONTACT_LATENT_ERROR_PROTOCOL_20261105_CODEX.md"
WORKER = ROOT / "experiments/pc_jepa_v22_contact_latent_error_worker_20261105_codex.py"


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads((SOURCE / "report.json").read_text())
    audit = json.loads((SOURCE / "audit.json").read_text())
    physical = json.loads((SOURCE / "physical.json").read_text())
    if (report["status"] != "completed" or audit["status"] != "passed"
            or audit["report_sha256"] != sha(SOURCE / "report.json")
            or audit["physical_sha256"] != sha(SOURCE / "physical.json")
            or physical["arrays_sha256"] != sha(SOURCE / "physical_arrays.npz")):
        raise ValueError("source audit mismatch")
    inputs = {"context.npz": SOURCE / "neural/predict/input/context.npz",
              "bank.npy": SOURCE / "neural/predict/input/bank.npy",
              "images.npz": SOURCE / "neural/encode/input/images.npz",
              "model_metadata.json": SOURCE / "neural/predict/input/model_metadata.json"}
    bindings = {str(p): sha(p) for p in [Path(__file__), WORKER, PROTOCOL, SOURCE / "report.json",
                                         SOURCE / "audit.json", SOURCE / "physical.json",
                                         SOURCE / "physical_arrays.npz", *inputs.values()]}
    OUT.mkdir()
    (OUT / "input").mkdir()
    (OUT / "output").mkdir()
    for name, source in inputs.items():
        shutil.copyfile(source, OUT / "input" / name)
        if sha(source) != sha(OUT / "input" / name):
            raise ValueError("input copy mismatch")
    with (OUT / "manifest.json").open("x") as handle:
        json.dump({"status": "prepared", "bindings": bindings}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Prepared fixed bank and evaluation labels without planner feedback.", flush=True)


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["status"] != "prepared" or any(sha(Path(p)) != h for p, h in manifest["bindings"].items()):
        raise ValueError("source or code changed")


def execute():
    verify()
    neural.launch(OUT, "experiments.pc_jepa_v22_contact_latent_error_worker_20261105_codex")


def summarize():
    verify()
    receipt = json.loads((OUT / "output/receipt.json").read_text())
    if (receipt["status"] != "completed" or receipt["worker_sha256"] != sha(WORKER)
            or receipt["latents_sha256"] != sha(OUT / "output/latents.npz")
            or any(receipt[key] != sha(OUT / "input" / name) for key, name in (
                ("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")))):
        raise ValueError("worker binding mismatch")
    with np.load(OUT / "output/latents.npz", allow_pickle=False) as data:
        predicted, actual, current, goal = [data[k].astype(np.float64) for k in ("predicted", "actual", "current", "goal")]
    if predicted.shape != (64, 256, 384) or actual.shape != predicted.shape or current.shape != (256, 384) or goal.shape != current.shape:
        raise ValueError("latent shapes")
    if not all(np.isfinite(x).all() for x in (predicted, actual, current, goal)):
        raise ValueError("nonfinite latent")
    pred_actual = ((predicted - actual) ** 2).mean(axis=(1, 2))
    persistence = ((current[None] - actual) ** 2).mean(axis=(1, 2))
    pred_goal = ((predicted - goal[None]) ** 2).mean(axis=(1, 2))
    actual_goal = ((actual - goal[None]) ** 2).mean(axis=(1, 2))
    p = json.loads((SOURCE / "neural/predict/output/decision.json").read_text())
    e = json.loads((SOURCE / "neural/encode/output/encoded.json").read_text())
    physical = json.loads((SOURCE / "physical.json").read_text())
    groups = []
    for prefix in range(4):
        start = 16 * prefix
        for name, indices in (("construction", np.arange(start + 1, start + 8)),
                              ("validation", np.arange(start + 8, start + 16))):
            angle = np.asarray([physical["rows"][int(i)]["metrics"]["angle_error_radians"] for i in indices])
            groups.append({"prefix": prefix, "group": name, "count": len(indices),
                           "mean_prediction_mse": float(pred_actual[indices].mean()),
                           "mean_persistence_mse": float(persistence[indices].mean()),
                           "prediction_beats_persistence_count": int(np.sum(pred_actual[indices] < persistence[indices])),
                           "mean_absolute_goal_score_error": float(np.mean(np.abs(pred_goal[indices]-actual_goal[indices]))),
                           "actual_goal_vs_angle": independent_order(actual_goal[indices], angle),
                           "predicted_goal_vs_angle": independent_order(pred_goal[indices], angle)})
    result = {"status": "completed", "worker_receipt_sha256": sha(OUT / "output/receipt.json"),
              "latents_sha256": sha(OUT / "output/latents.npz"),
              "max_predicted_score_difference_from_original": float(np.max(np.abs(pred_goal - p["decision"]["scores"]))),
              "max_actual_score_difference_from_original": float(np.max(np.abs(actual_goal - e["scores"]))),
              "groups": groups,
              "rows": [{"index": i, "prediction_mse": float(pred_actual[i]), "persistence_mse": float(persistence[i]),
                        "predicted_goal_mse": float(pred_goal[i]), "actual_goal_mse": float(actual_goal[i]),
                        "physical": physical["rows"][i]["metrics"]} for i in range(64)],
              "scope": "single exposed constructed root, evaluator-only privileged candidates"}
    with (OUT / "report.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "groups": groups,
                      "max_predicted_score_difference_from_original": result["max_predicted_score_difference_from_original"],
                      "max_actual_score_difference_from_original": result["max_actual_score_difference_from_original"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "execute", "summarize"))
    args = parser.parse_args()
    {"prepare": prepare, "execute": execute, "summarize": summarize}[args.phase]()


if __name__ == "__main__":
    main()
