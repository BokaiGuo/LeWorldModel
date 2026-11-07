"""Model-only score-margin gate; assess sealed pairs using historical physical traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_effect_bank_three_way_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash

PROTOCOL = source.src.prior.ROOT.parent / "docs/PC_EFFECT_MARGIN_GATE_PROTOCOL_20261107_CODEX.md"
BASE = source.src.prior.ROOT / "results/pc_effect_margin_gate_20261107_codex"
PLAN, RESULT = [Path(str(BASE) + suffix) for suffix in ("_plan.json", "_result.json")]
CALIBRATION, EVALUATION = tuple(range(10, 20)), tuple(range(20, 50))


def margins(cost):
    cost = np.asarray(cost)
    if cost.shape != (8, 9) or not np.isfinite(cost).all():
        raise ValueError("invalid costs")
    return cost[:, :1] - cost[:, 1:]


def choose(cost, threshold):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid threshold")
    delta = margins(cost)
    return {"direct": (delta > 0), "gated": (delta > threshold),
            "reference": np.zeros((8, 8), dtype=bool)}


def predict():
    source.src.verify()
    if PLAN.exists():
        raise FileExistsError(PLAN)
    meta = json.loads(source.META.read_text())
    for key, path in (("pred_sha256", source.PRED), ("runner_sha256", source.__file__),
                      ("protocol_sha256", source.PROTOCOL), ("source_seal_sha256", source.src.SEAL)):
        if meta[key] != file_hash(path):
            raise ValueError(f"source mismatch: {key}")
    rows = []
    with np.load(source.PRED, allow_pickle=False) as pred:
        threshold = float(np.median(np.concatenate([np.abs(margins(pred[f"r{root}_cost"])).ravel()
                                                   for root in CALIBRATION])))
        for root in EVALUATION:
            arms = choose(pred[f"r{root}_cost"], threshold)
            rows.append({"root": root, "choose_probe": {key: value.tolist() for key, value in arms.items()}})
    with PLAN.open("x") as stream:
        json.dump({"threshold": threshold, "calibration_roots": CALIBRATION, "evaluation_roots": EVALUATION,
                   "rows": rows, "prediction_sha256": file_hash(source.PRED), "source_meta_sha256": file_hash(source.META),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "model-only selection; evaluator outcomes not read"},
                  stream, indent=2, allow_nan=False)


def outcome(distance, term, select):
    distance, term, select = map(np.asarray, (distance, term, select))
    if distance.shape != (8, 9) or term.shape != (8, 9, 25) or select.shape != (8, 8):
        raise ValueError("invalid shapes")
    if not np.isfinite(distance).all() or term.dtype != np.bool_ or select.dtype != np.bool_:
        raise ValueError("invalid data")
    return {"distance": np.where(select, distance[:, 1:], distance[:, :1]),
            "ever_success": np.where(select, term[:, 1:].any(-1), term[:, :1].any(-1))}


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    for key, path in (("prediction_sha256", source.PRED), ("source_meta_sha256", source.META),
                      ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if plan[key] != file_hash(path):
            raise ValueError(f"binding mismatch: {key}")
    meta = json.loads(source.META.read_text())
    if meta["source_physical_sha256"] != file_hash(source.src.PHYS):
        raise ValueError("physical artifact drift")
    rows = []
    with np.load(source.PRED, allow_pickle=False) as pred, np.load(source.src.PHYS, allow_pickle=False) as physical:
        for row in plan["rows"]:
            root = row["root"]
            choices = choose(pred[f"r{root}_cost"], plan["threshold"])
            if any(not np.array_equal(choices[k], np.asarray(row["choose_probe"][k])) for k in choices):
                raise ValueError("sealed choice mismatch")
            d = physical[f"r{root}_distance"][:, :, -1]
            term = physical[f"r{root}_term"]
            arms = {}
            for key, selection in choices.items():
                result = outcome(d, term, selection)
                arms[key] = {"distance": float(result["distance"].mean()),
                             "ever_success": float(result["ever_success"].mean()),
                             "probe_count": int(selection.sum()),
                             "distance_per_pair": result["distance"].tolist(),
                             "success_per_pair": result["ever_success"].tolist()}
            rows.append({"root": root, "arms": arms})
    if [row["root"] for row in rows] != list(EVALUATION):
        raise ValueError("incorrect roots")
    summary = {}
    for arm in ("direct", "gated", "reference"):
        summary[arm] = {metric: float(np.mean([row["arms"][arm][metric] for row in rows]))
                        for metric in ("distance", "ever_success", "probe_count")}
    summary["distance_improvement"] = summary["direct"]["distance"] - summary["gated"]["distance"]
    summary["success_difference"] = summary["gated"]["ever_success"] - summary["direct"]["ever_success"]
    summary["improved_roots"] = sum(row["arms"]["gated"]["distance"] < row["arms"]["direct"]["distance"] for row in rows)
    summary["development_exit"] = bool(summary["distance_improvement"] >= .5 and
                                       summary["improved_roots"] >= 18 and summary["success_difference"] >= -.01)
    with RESULT.open("x") as stream:
        json.dump({"summary": summary, "rows": rows, "plan_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(source.src.PHYS), "runner_sha256": file_hash(__file__),
                   "scope": "exposed development, 64 paired binary comparisons per task"}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "evaluate"):
        raise SystemExit("expected predict|evaluate")
    {"predict": predict, "evaluate": evaluate}[sys.argv[1]]()
