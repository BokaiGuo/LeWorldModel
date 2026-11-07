"""Recompute exact costs and aggregate paired temporal-tail development results."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior
from experiments import pc_exact_temporal_tail_dev_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_exact_temporal_tail_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    if result["dependencies"] != {p: prior.sha(p) for p in result["dependencies"]}:
        raise ValueError("run dependencies changed")
    source = json.loads(run.SOURCE.read_text())
    prior_data = json.loads(prior.OUTPUT.read_text())
    rows = []
    for row, baseline, original in zip(result["rows"], source["rows"], prior_data["rows"], strict=True):
        if row["root"] != baseline["root"] or row["root"] != original["root"]:
            raise ValueError("root mismatch")
        if row["selected_lookahead_prefixes"] != original["selected"]["lookahead"]:
            raise ValueError("selected prefix mismatch")
        pool = np.asarray(original["pool"])
        values = []
        for i, search in enumerate(row["temporal"]):
            chains = np.asarray(search["actions"])
            costs = np.asarray(search["costs"])
            if chains.shape != (384, 15, 2) or costs.shape != (384,) or np.abs(chains).max() > 1:
                raise ValueError("invalid candidate bank")
            np.testing.assert_array_equal(chains[0], pool[i])
            np.testing.assert_array_equal(chains[:, :5], np.broadcast_to(pool[i, :5], (384, 5, 2)))
            actual = np.linalg.norm(prior.exact.endpoints(row["task"], chains) - row["task"]["goal"], axis=1)
            np.testing.assert_allclose(actual, costs, atol=1e-12, rtol=0)
            if search["best"] != int(np.argmin(costs)):
                raise ValueError("best candidate mismatch")
            values.append(float(costs[search["best"]]))
        base = [float(x["costs"][x["best"]]) for x in baseline["pool_longer"]]
        selected = row["selected_lookahead_prefixes"]
        rows.append({"root": row["root"], "baseline_oracle": min(base),
                     "temporal_oracle": min(values), "baseline_lookahead": min(base[i] for i in selected),
                     "temporal_lookahead": min(values[i] for i in selected),
                     "baseline_by_prefix": base, "temporal_by_prefix": values})
    summary = {}
    for name in ("baseline_oracle", "temporal_oracle", "baseline_lookahead", "temporal_lookahead"):
        scores = np.asarray([r[name] for r in rows])
        summary[name] = {"mean_distance": float(scores.mean()), "success_count": int((scores < .05).sum())}
    baseline = np.asarray([r["baseline_oracle"] for r in rows])
    temporal = np.asarray([r["temporal_oracle"] for r in rows])
    summary["oracle_strictly_better_roots"] = int((temporal < baseline - 1e-12).sum())
    summary["oracle_mean_reduction"] = float((baseline - temporal).mean())
    summary["development_exit_met"] = bool(summary["oracle_mean_reduction"] >= .02
        and summary["oracle_strictly_better_roots"] >= 14
        and summary["temporal_lookahead"]["mean_distance"] <= summary["baseline_lookahead"]["mean_distance"])
    output = {"rows": rows, "summary": summary, "run_sha256": prior.sha(run.OUTPUT),
              "audit_sha256": prior.sha(__file__), "recomputed_candidate_count": 24 * 8 * 384,
              "scope": "exposed exact dynamics development, not independent confirmation"}
    with OUTPUT.open("x") as file:
        json.dump(output, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
