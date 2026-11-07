"""Post-hoc, same-packet decomposition of latent and physical ordering."""
import json

import numpy as np

from experiments import pc_tworoom_wide_offset_alignment_20261107_codex as src
from experiments.pc_input_permissions import file_hash

OUTPUT = src.ROOT / "results/pc_tworoom_wide_offset_failure_decomposition_20261107_codex.json"
BOOTSTRAP_SEED = 20261107


def pair_categories(predicted, encoded, physical, tol=1e-6):
    arrays = [np.asarray(x, dtype=float) for x in (predicted, encoded, physical)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("invalid paired data")
    counts = np.zeros(4, dtype=int)
    ties = 0
    for i in range(len(arrays[0])):
        for j in range(i + 1, len(arrays[0])):
            dp, de, dt = (a[i] - a[j] for a in arrays)
            if abs(dt) <= tol or abs(dp) <= tol or abs(de) <= tol:
                ties += 1
                continue
            p = np.sign(dp) == np.sign(dt)
            e = np.sign(de) == np.sign(dt)
            counts[0 if p and e else 1 if not p and e else 2 if p and not e else 3] += 1
    return counts, ties


def summarize(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("invalid root means")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "root_bootstrap_95": np.quantile(samples, [.025, .975]).tolist(),
            "roots": len(x)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = json.loads(src.PLAN.read_text())
    result = json.loads(src.RESULT.read_text())
    for expected, path in ((seal["pred_sha256"], src.PRED), (result["plan_sha256"], src.PLAN),
                           (result["actual_sha256"], src.ACTUAL)):
        if expected != file_hash(path):
            raise ValueError(f"input drift: {path}")
    rows = []
    print("Compare prediction and encoded-goal ordering against physical distance at identical packet times.")
    with np.load(src.PRED, allow_pickle=False) as pred, np.load(src.ACTUAL, allow_pickle=False) as actual:
        for i, (plan, trace) in enumerate(zip(seal["rows"], result["rows"], strict=True)):
            if plan["seed"] != trace["seed"]:
                raise ValueError("root mismatch")
            p, z, goal = pred[f"pred_{i}"], actual[f"actual_{i}"], pred[f"goal_{i}"]
            if p.shape != z.shape or p.shape[:2] != (13, 5) or goal.shape != p.shape[2:]:
                raise ValueError("representation clock mismatch")
            cp, cz = ((p - goal) ** 2).sum(-1), ((z - goal) ** 2).sum(-1)
            np.testing.assert_allclose(cp, plan["costs"], atol=1e-3, rtol=1e-5)
            np.testing.assert_allclose(cz, trace["encoded_costs"], atol=1e-3, rtol=1e-5)
            distances = np.array([[item["distance"] for item in chain] for chain in trace["chains"]])
            if distances.shape != (13, 25) or not np.isfinite(distances).all():
                raise ValueError("physical clock mismatch")
            packet_rows = []
            for k in range(5):
                counts, ties = pair_categories(cp[:, k], cz[:, k], distances[:, 5 * k + 4])
                denominator = int(counts.sum())
                packet_rows.append({"counts": counts.tolist(), "ties": ties,
                                    "fractions": (counts / denominator).tolist() if denominator else None,
                                    "feature_mse": float(np.mean((p[:, k] - z[:, k]) ** 2))})
            rows.append({"seed": plan["seed"], "packets": packet_rows})
    if len(rows) != 40:
        raise ValueError("unexpected panel")
    summaries = []
    for k in range(5):
        fractions = np.asarray([r["packets"][k]["fractions"] for r in rows
                                if r["packets"][k]["fractions"] is not None])
        summaries.append({"environment_step": 5 * (k + 1),
                          "both_match": summarize(fractions[:, 0]) if len(fractions) else None,
                          "prediction_only_error": summarize(fractions[:, 1]) if len(fractions) else None,
                          "encoding_only_error": summarize(fractions[:, 2]) if len(fractions) else None,
                          "both_error": summarize(fractions[:, 3]) if len(fractions) else None,
                          "incomparable_roots": len(rows) - len(fractions),
                          "feature_mse": summarize([r["packets"][k]["feature_mse"] for r in rows]),
                          "ties": sum(r["packets"][k]["ties"] for r in rows)})
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-hoc exposed tasks; ties excluded per packet; no causal identification",
                   "sha256": {str(p): file_hash(p) for p in (src.PLAN, src.RESULT, src.PRED, src.ACTUAL, __file__)},
                   "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": 10000, "unit": "root"},
                   "categories_order": ["both_match", "prediction_only_error", "encoding_only_error", "both_error"],
                   "summaries": summaries, "rows": rows}, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
