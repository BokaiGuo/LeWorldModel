"""Within-prefix evaluation of sealed, exposed TwoRoom candidate trajectories."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / "docs"
PROTOCOL = DOCS / "PC_TWOROOM_WITHIN_PREFIX_PAIRWISE_POSTHOC_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_tworoom_block_heldout_effect_20261107_codex"
TEMP = ROOT / "results/pc_tworoom_block_b_temporal_alignment_20261107_codex"
PLAN = Path(str(BASE) + "_plan.json")
POOL = Path(str(BASE) + "_pool.npz")
PHYSICAL = Path(str(BASE) + "_physical.json")
RESULT = Path(str(TEMP) + "_result.json")
FEATURES = Path(str(TEMP) + "_features.npz")
PRED = Path(str(TEMP) + "_pred.npz")
TEMP_PHYSICAL = Path(str(TEMP) + "_physical.json")
OUT = ROOT / "results/pc_tworoom_within_prefix_pairwise_posthoc_20261107_codex.json"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_bindings():
    plan = json.loads(PLAN.read_text())
    physical = json.loads(PHYSICAL.read_text())
    temporal = json.loads(RESULT.read_text())
    for actual, path in ((plan["pool_sha256"], POOL), (physical["plan_sha256"], PLAN),
                         (physical["pool_sha256"], POOL), (temporal["features_sha256"], FEATURES),
                         (temporal["pred_sha256"], PRED), (temporal["physical_sha256"], TEMP_PHYSICAL)):
        if actual != digest(path):
            raise ValueError(f"input binding mismatch: {path}")
    tphys = json.loads(TEMP_PHYSICAL.read_text())
    if tphys["old_physical_sha256"] != digest(PHYSICAL):
        raise ValueError("temporal physical source mismatch")
    return plan, physical


def pair_counts(x, y, allowed):
    x, y, allowed = np.asarray(x), np.asarray(y), np.asarray(allowed)
    if x.shape != (8, 8) or y.shape != x.shape or allowed.shape != (8,):
        raise ValueError("expected eight prefixes and eight tails")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or allowed.dtype != np.bool_:
        raise ValueError("nonfinite score or invalid mask")
    counts = dict(agree=0, disagree=0, x_tie=0, y_tie=0, both_tie=0)
    for row in np.flatnonzero(allowed):
        for i in range(8):
            for j in range(i + 1, 8):
                a, b = np.sign(x[row, i] - x[row, j]), np.sign(y[row, i] - y[row, j])
                key = ("both_tie" if a == 0 and b == 0 else "x_tie" if a == 0
                       else "y_tie" if b == 0 else "agree" if a == b else "disagree")
                counts[key] += 1
    n = counts["agree"] + counts["disagree"]
    counts["accuracy"] = counts["agree"] / n if n else None
    return counts


def validate(actions, outcomes):
    actions = np.asarray(actions)
    if actions.shape != (8, 8, 25, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
        raise ValueError("invalid action pool")
    if not np.array_equal(actions[:, :, :10], np.broadcast_to(actions[:, :1, :10], (8, 8, 10, 2))):
        raise ValueError("prefix differs within a row")
    if not np.array_equal(actions[:, :, 10:], np.broadcast_to(actions[:1, :, 10:], (8, 8, 15, 2))):
        raise ValueError("tail differs within a column")
    if len(outcomes) != 64:
        raise ValueError("incomplete outcomes")
    prefix = []
    for row in range(8):
        traces = [outcomes[row * 8 + col]["trace"] for col in range(8)]
        if any(len(t) != 25 for t in traces) or any(t[:10] != traces[0][:10] for t in traces[1:]):
            raise ValueError("physical prefix trajectories differ")
        pre = any(v["terminated"] for v in traces[0][:10])
        for col, trace in enumerate(traces):
            o = outcomes[row * 8 + col]
            if o["prefix_success"] != pre or o["ever"] != any(v["terminated"] for v in trace):
                raise ValueError("inconsistent success label")
        prefix.append(pre)
    return np.array(prefix, dtype=bool)


def evaluate(actions, outcomes, predicted, encoded):
    pre = validate(actions, outcomes)
    predicted, encoded = np.asarray(predicted), np.asarray(encoded)
    if predicted.shape != (64, 5) or encoded.shape != (64, 5):
        raise ValueError("invalid temporal score shape")
    distances = np.array([[o["trace"][i]["distance"] for i in (4, 9, 14, 19, 24)]
                          for o in outcomes])
    success = np.array([o["ever"] for o in outcomes], dtype=float).reshape(8, 8)
    subsets = {"all": np.ones(8, dtype=bool), "prefix_not_successful": ~pre}
    out = {"prefix_success_count": int(pre.sum()), "subsets": {}}
    for name, mask in subsets.items():
        entry = {"prefix_count": int(mask.sum()), "steps": {}}
        for step, time in ((15, 2), (20, 3), (25, 4)):
            p, e, d = (a[:, time].reshape(8, 8) for a in (predicted, encoded, distances))
            entry["steps"][str(step)] = {
                "prediction_vs_encoding": pair_counts(p, e, mask),
                "prediction_vs_distance": pair_counts(p, d, mask),
                "encoding_vs_distance": pair_counts(e, d, mask),
                "prediction_vs_ever_success": pair_counts(p, -success, mask),
                "encoding_vs_ever_success": pair_counts(e, -success, mask),
            }
        out["subsets"][name] = entry
    return out


def main():
    plan, physical = check_bindings()
    rows = []
    with np.load(POOL, allow_pickle=False) as pool, np.load(FEATURES, allow_pickle=False) as feat:
        for root in range(10):
            p, o = plan["rows"][root], physical["rows"][root]
            if p["root"] != root or o["root"] != root or p["row"] != o["row"]:
                raise ValueError("root identity mismatch")
            result = evaluate(pool[f"r{root}_b_actions"], o["outcomes"],
                              feat[f"r{root}_predicted_cost"], feat[f"r{root}_encoded_cost"])
            rows.append({"root": root, "row": p["row"], **result})
    summary = {}
    for subset in ("all", "prefix_not_successful"):
        summary[subset] = {}
        for step in ("15", "20", "25"):
            summary[subset][step] = {}
            for key in ("prediction_vs_encoding", "prediction_vs_distance", "encoding_vs_distance",
                        "prediction_vs_ever_success", "encoding_vs_ever_success"):
                items = [r["subsets"][subset]["steps"][step][key] for r in rows]
                rates = [x["accuracy"] for x in items if x["accuracy"] is not None]
                summary[subset][step][key] = {
                    "equal_root_accuracy": float(np.mean(rates)) if rates else None,
                    "valid_roots": len(rates),
                    "counts": {k: sum(x[k] for x in items) for k in
                               ("agree", "disagree", "x_tie", "y_tie", "both_tie")}}
    data = {"scope": "posthoc exposed development; evaluation truth excluded from planner",
            "sha256": {str(p): digest(p) for p in (PLAN, POOL, PHYSICAL, RESULT, FEATURES, PRED,
                                                    TEMP_PHYSICAL, PROTOCOL, Path(__file__))},
            "rows": rows, "summary": summary}
    with OUT.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
