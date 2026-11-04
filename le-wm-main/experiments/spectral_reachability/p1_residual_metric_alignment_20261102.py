"""Compare pre-existing latent metrics against paired physical outcomes.

This uses post-outcome development data and cannot be used for action selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_residual_cp5_feedback_20261102 as planner
from . import p1_residual_cp5_physical_20261102 as physical
from . import p1_residual_cp5_saved_20261102 as saved

ROOT = planner.ROOT
AUDIT = ROOT / "results/p1_residual_cp5_prediction_audit_20261102_run01.json"
AUDIT_SHA = "2671ffc19e41ccd91aaf3f62f84d239f390d789dc2a2031f4946966e8f1d3452"
AUTHORITY = "P1_RESIDUAL_METRIC_DIAGNOSTIC_CUDA"


def pairwise(rows, predicted, actual, physical_value):
    """Lower latent distances and lower physical errors are preferred."""
    pairs = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            diffs = [float(left[key]) - float(right[key]) for key in
                     (predicted, actual, physical_value)]
            if not np.isfinite(diffs).all():
                raise ValueError("nonfinite pair")
            signs = [int(np.sign(x)) for x in diffs]
            pairs.append({"left": left["arm"], "right": right["arm"], "differences": diffs,
                          "predicted_actual_inversion": signs[0] * signs[1] < 0,
                          "actual_physical_inversion": signs[1] * signs[2] < 0,
                          "contains_tie": 0 in signs})
    return pairs


def run(output, authority):
    planner.require(authority == AUTHORITY and not output.exists(), "authority/output")
    verified = saved.analyze()
    audit = physical.read(AUDIT, AUDIT_SHA)
    planner.require(audit["physical_report_sha256"] == saved.REPORT_SHA
                    and len(audit["rows"]) == len(verified["rows"]) == 8, "paired audit")
    source = planner.protocol()
    planner.require(audit["model_before"] == source["model_hash"]
                    and audit["model_before"] == audit["model_after"], "model identity")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(source["policy"], device=device).eval().requires_grad_(False)
    hash_model = planner.engine.base.blind.host.signal.model_hash
    before = hash_model(model)
    planner.require(before == source["model_hash"], "checkpoint identity")
    _, task, goal_rgb, _, _ = planner.source_inputs()
    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        metric, metadata = planner.metric.build(model, goal_rgb, device)
    goal = np.asarray(task["goal"], dtype=np.float64)
    planner.require(np.allclose(metadata["goal_latent"], goal, rtol=1e-5, atol=1e-6), "goal binding")
    physical_by_arm = {row["arm"]: row for row in verified["rows"]}
    rows = []
    for row in audit["rows"]:
        vectors = row["latent_vectors"]
        actual = np.asarray(vectors["actual_second"], dtype=np.float64)
        predicted = np.asarray(vectors["predicted_open_second"], dtype=np.float64)
        planner.require(actual.shape == predicted.shape == goal.shape == (192,)
                        and np.isfinite(actual).all() and np.isfinite(predicted).all(), "latent vectors")
        p = physical_by_arm[row["arm"]]
        def qscore(z):
            return float(np.mean((metric @ (z-goal)) ** 2))
        rows.append({"arm": row["arm"], "group": row["metric"],
                     "predicted_latent": float(np.mean((predicted-goal)**2)),
                     "actual_latent": float(np.mean((actual-goal)**2)),
                     "predicted_q": qscore(predicted), "actual_q": qscore(actual),
                     "object_position_error": p["terminal"]["object_position_error"],
                     "object_angle_error": p["terminal"]["object_angle_error"],
                     "negative_reward": -p["terminal_reward"]})
    groups = {}
    for group in ("latent", "goal_tangent"):
        selected = [r for r in rows if r["group"] == group]
        planner.require(len(selected) == 4, "group roster")
        groups[group] = {f"{m}_vs_{outcome}": pairwise(selected, f"predicted_{m}", f"actual_{m}", outcome)
                         for m in ("latent", "q")
                         for outcome in ("object_position_error", "object_angle_error", "negative_reward")}
    torch.cuda.synchronize()
    after = hash_model(model)
    planner.require(before == after, "frozen model changed")
    return physical.write_new(output, {"schema": "p1_residual_metric_alignment.v1", "rows": rows,
        "pairs": groups, "physical_report_sha256": saved.REPORT_SHA, "prediction_sha256": AUDIT_SHA,
        "source_sha256": physical.sha(__file__), "model_before": before, "model_after": after, "host": host,
        "goal_tangent_metric": metadata, "classification": "posthoc_development_no_selection",
        "independent_confirmation": False, "P1_admission": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authority", required=True)
    args = parser.parse_args()
    with planner.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        print(json.dumps({"sha256": run(args.output, args.authority)}))
