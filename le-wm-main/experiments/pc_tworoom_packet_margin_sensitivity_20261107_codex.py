"""Post-hoc sensitivity to physically meaningful pair separation."""
import json

import numpy as np

from experiments import pc_tworoom_wide_offset_failure_decomposition_20261107_codex as audit
from experiments.pc_input_permissions import file_hash

src = audit.src
OUTPUT = src.ROOT / "results/pc_tworoom_packet_margin_sensitivity_20261107_codex.json"


def categories(pred, encoded, physical, margin):
    pred, encoded, physical = (np.asarray(a, dtype=float) for a in (pred, encoded, physical))
    if (pred.ndim != 1 or pred.shape != encoded.shape or pred.shape != physical.shape
            or not np.isfinite(np.r_[pred, encoded, physical, margin]).all() or margin < 0):
        raise ValueError("invalid pair arrays")
    counts = np.zeros(4, dtype=int)
    excluded = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            dp, de, dt = pred[i] - pred[j], encoded[i] - encoded[j], physical[i] - physical[j]
            if abs(dt) <= margin or abs(dp) <= 1e-6 or abs(de) <= 1e-6:
                excluded += 1
                continue
            p, e = np.sign(dp) == np.sign(dt), np.sign(de) == np.sign(dt)
            counts[0 if p and e else 1 if not p and e else 2 if p and not e else 3] += 1
    return counts, excluded


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    previous = json.loads(audit.OUTPUT.read_text())
    for path, expected in previous["sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"previous analysis drift: {path}")
    plan, physical = json.loads(src.PLAN.read_text()), json.loads(src.RESULT.read_text())
    rows = []
    print("Assess sensitivity of packet-level ordering to 1- and 5-pixel physical separation.", flush=True)
    with np.load(src.PRED, allow_pickle=False) as pred, np.load(src.ACTUAL, allow_pickle=False) as actual:
        for i, (p_row, r_row) in enumerate(zip(plan["rows"], physical["rows"], strict=True)):
            if p_row["seed"] != r_row["seed"]:
                raise ValueError("root mismatch")
            goal = pred[f"goal_{i}"]
            cp = ((pred[f"pred_{i}"] - goal) ** 2).sum(-1)
            cz = ((actual[f"actual_{i}"] - goal) ** 2).sum(-1)
            distances = np.array([[step["distance"] for step in chain] for chain in r_row["chains"]])
            if cp.shape != (13, 5) or cz.shape != (13, 5) or distances.shape != (13, 25):
                raise ValueError("clock mismatch")
            packets = []
            for k in range(5):
                by_margin = {}
                for margin in (1., 5.):
                    counts, excluded = categories(cp[:, k], cz[:, k], distances[:, 5*k+4], margin)
                    by_margin[str(int(margin))] = {"counts": counts.tolist(), "excluded": excluded,
                                                  "fractions": (counts / counts.sum()).tolist()
                                                  if counts.sum() else None}
                packets.append(by_margin)
            rows.append({"seed": p_row["seed"], "packets": packets})
    summaries = []
    for k in range(5):
        for margin in ("1", "5"):
            valid = [r["packets"][k][margin]["fractions"] for r in rows
                     if r["packets"][k][margin]["fractions"] is not None]
            fractions = np.asarray(valid)
            summaries.append({"step": 5 * (k + 1), "margin": int(margin),
                              "eligible_roots": len(valid),
                              "category_means": fractions.mean(axis=0).tolist() if len(valid) else None,
                              "excluded_pairs": sum(r["packets"][k][margin]["excluded"] for r in rows)})
    with OUTPUT.open("x") as f:
        json.dump({"scope": "post-hoc sensitivity; physical distance margins in environment position units",
                   "previous_sha256": file_hash(audit.OUTPUT), "runner_sha256": file_hash(__file__),
                   "summaries": summaries, "rows": rows}, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
