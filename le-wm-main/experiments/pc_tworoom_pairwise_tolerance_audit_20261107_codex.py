"""Sensitivity of previously published within-prefix rankings to near ties."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_within_prefix_pairwise_posthoc_20261107_codex as base

OUT = base.ROOT / "results/pc_tworoom_pairwise_tolerance_audit_20261107_codex.json"
# Two independent values can each vary by the earlier 1e-3 score-replay tolerance.
# The physical replay tolerance is 1e-5 per distance. Neither is a uniform error bound.
THRESHOLDS = ((0.0, 0.0), (0.002, 0.00002), (0.01, 0.0001))


def counts(x, y, allowed, x_tol, y_tol):
    x, y, allowed = np.asarray(x), np.asarray(y), np.asarray(allowed)
    if (x.shape != (8, 8) or y.shape != x.shape or allowed.shape != (8,)
            or allowed.dtype != np.bool_ or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite([x_tol, y_tol]).all()
            or min(x_tol, y_tol) < 0):
        raise ValueError("invalid inputs")
    out = dict(agree=0, disagree=0, x_tie=0, y_tie=0, both_tie=0)
    for row in np.flatnonzero(allowed):
        for i in range(8):
            for j in range(i + 1, 8):
                dx, dy = float(x[row, i] - x[row, j]), float(y[row, i] - y[row, j])
                xt, yt = abs(dx) <= x_tol, abs(dy) <= y_tol
                key = ("both_tie" if xt and yt else "x_tie" if xt else
                       "y_tie" if yt else "agree" if np.sign(dx) == np.sign(dy)
                       else "disagree")
                out[key] += 1
    n = out["agree"] + out["disagree"]
    out["accuracy"] = out["agree"] / n if n else None
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = base.check_bindings()
    rows = []
    with np.load(base.POOL, allow_pickle=False) as pool, np.load(base.FEATURES, allow_pickle=False) as features:
        for root in range(10):
            p, record = plan["rows"][root], physical["rows"][root]
            if p["root"] != root or record["root"] != root or p["row"] != record["row"]:
                raise ValueError("root identity mismatch")
            outcomes = record["outcomes"]
            eligible = ~base.validate(pool[f"r{root}_b_actions"], outcomes)
            predicted = features[f"r{root}_predicted_cost"]
            encoded = features[f"r{root}_encoded_cost"]
            if predicted.shape != (64, 5) or encoded.shape != (64, 5):
                raise ValueError("wrong score shape")
            distances = np.array([[o["trace"][i]["distance"] for i in (4, 9, 14, 19, 24)]
                                  for o in outcomes])
            cases = {}
            for t in (2, 3, 4):
                v = [arr[:, t].reshape(8, 8) for arr in (predicted, encoded, distances)]
                cases[str((t + 1) * 5)] = {}
                for score_tol, dist_tol in THRESHOLDS:
                    name = f"{score_tol:g}/{dist_tol:g}"
                    cases[str((t + 1) * 5)][name] = {
                        "prediction_vs_encoding": counts(v[0], v[1], eligible, score_tol, score_tol),
                        "prediction_vs_distance": counts(v[0], v[2], eligible, score_tol, dist_tol),
                        "encoding_vs_distance": counts(v[1], v[2], eligible, score_tol, dist_tol)}
                    if score_tol == dist_tol == 0:
                        for label, left, right in (("prediction_vs_encoding", 0, 1),
                                                   ("prediction_vs_distance", 0, 2),
                                                   ("encoding_vs_distance", 1, 2)):
                            if cases[str((t + 1) * 5)][name][label] != base.pair_counts(v[left], v[right], eligible):
                                raise ValueError("original metric reproduction failed")
            rows.append({"root": root, "row": p["row"], "eligible_prefixes": int(eligible.sum()), "steps": cases})
    if len(rows) != 10:
        raise ValueError("incomplete roots")
    summary = {}
    for step in ("15", "20", "25"):
        summary[step] = {}
        for st, dt in THRESHOLDS:
            name = f"{st:g}/{dt:g}"
            summary[step][name] = {}
            for label in ("prediction_vs_encoding", "prediction_vs_distance", "encoding_vs_distance"):
                values = [r["steps"][step][name][label] for r in rows]
                valid = [v["accuracy"] for v in values if v["accuracy"] is not None]
                summary[step][name][label] = {
                    "root_mean": float(np.mean(valid)) if valid else None,
                    "valid_roots": len(valid),
                    "counts": {k: sum(v[k] for v in values)
                               for k in ("agree", "disagree", "x_tie", "y_tie", "both_tie")}}
    sources = (base.PLAN, base.POOL, base.PHYSICAL, base.RESULT, base.FEATURES,
               base.PRED, base.TEMP_PHYSICAL, Path(__file__))
    output = {"scope": "posthoc exposed development; tolerances are sensitivity thresholds, not certified error bounds",
              "thresholds_score_distance": THRESHOLDS, "rows": rows, "summary": summary,
              "sha256": {str(path): base.digest(path) for path in sources}}
    with OUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
