"""Post-hoc root-level alignment of sealed model and physical action effects."""
import json
from itertools import combinations

import numpy as np

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments.pc_input_permissions import file_hash

OUTPUT = base.ROOT / "results/pc_tworoom_effect_vector_physical_audit_20261107_codex.json"


def pairwise_accuracy(model, physical, tolerance=1e-6):
    """Comparable pairs only; prediction ties contribute one half."""
    model, physical = np.asarray(model, float), np.asarray(physical, float)
    if model.ndim != 1 or physical.shape != model.shape or not np.isfinite(model).all() or not np.isfinite(physical).all():
        raise ValueError("invalid effects")
    scores = []
    for i, j in combinations(range(len(model)), 2):
        truth = physical[i] - physical[j]
        if abs(truth) <= tolerance:
            continue
        pred = model[i] - model[j]
        scores.append(.5 if abs(pred) <= tolerance else float(pred * truth > 0))
    return None if not scores else sum(scores) / len(scores)


def audit_row(plan_row, replay_row):
    cost = np.asarray(plan_row["costs"], dtype=float)
    physical = np.asarray(replay_row["physical"], dtype=float)
    if cost.shape != (13, 9) or physical.shape != (13, 9, 25, 3):
        raise ValueError("invalid matrix or trajectory")
    # A probes are columns 0:4; same-prefix zero tail is column 4.
    model_effect = cost[:, 4, None] - cost[:, :4]
    physical_effect = physical[:, 4, -1, 0, None] - physical[:, :4, -1, 0]
    # Exact identical prefixes count once; prevent duplicate slots from changing averages.
    prefixes = np.asarray(plan_row["prefixes"], np.float32)
    reps = []
    for i, prefix in enumerate(prefixes):
        if not any(np.array_equal(prefix, prefixes[j]) for j in reps):
            reps.append(i)
    scores = [pairwise_accuracy(model_effect[i], physical_effect[i]) for i in reps]
    valid = [s for s in scores if s is not None]
    selected = plan_row["selected"]["descriptor"][0]
    return {"seed": plan_row["seed"], "representatives": reps,
            "prefix_accuracy": scores, "task_mean": None if not valid else float(np.mean(valid)),
            "selected_prefix_accuracy": pairwise_accuracy(model_effect[selected], physical_effect[selected]),
            "model_effect": model_effect.tolist(), "physical_effect_distance": physical_effect.tolist(),
            "a_success": physical[:, :4, :, 1].astype(bool).any(axis=2).tolist(),
            "prefix_already_success": physical[:, :4, :10, 1].astype(bool).any(axis=2).tolist()}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, replay = json.loads(base.PLAN.read_text()), json.loads(base.RESULT.read_text())
    if replay["plan_sha256"] != file_hash(base.PLAN) or replay["truth_sha256"] != file_hash(base.PANEL.TRUTH):
        raise ValueError("replay provenance drift")
    for key, path in (("public_sha256", base.PANEL.PUBLIC), ("protocol_sha256", base.PROTOCOL),
                      ("runner_sha256", base.__file__), ("wide_sha256", base.wide.__file__),
                      ("repair_sha256", base.wide.repair.__file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    if len(plan["rows"]) != 40 or len(replay["rows"]) != 40:
        raise ValueError("root count drift")
    rows = []
    for p, r in zip(plan["rows"], replay["rows"]):
        if p["seed"] != r["seed"]:
            raise ValueError("root mismatch")
        rows.append(audit_row(p, r))
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-hoc exposed tasks; endpoint physical distance is diagnostic only",
                   "plan_sha256": file_hash(base.PLAN), "physical_sha256": file_hash(base.RESULT),
                   "runner_sha256": file_hash(__file__), "rows": rows}, f, allow_nan=False)
    print(f"Audited {len(rows)} roots; no planner choices were changed.")


if __name__ == "__main__":
    main()
