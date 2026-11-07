"""Reconstruct sealed candidate lineage without further model or physical calls."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_pointmaze_goal25_retention_cem_20261107_codex as source
from experiments.pc_input_permissions import file_hash

BASE = source.ROOT / "results/pc_pointmaze_goal25_selection_audit_20261107_codex"
CANDIDATES = Path(str(BASE) + "_cem_candidates.npz")
OUT = Path(str(BASE) + "_result.json")


def reconstruct_cem(record):
    rng = np.random.default_rng(record["seed"])
    mean, std = np.zeros((25, 2), np.float32), np.full((25, 2), .5, np.float32)
    pool, costs = [], []
    if len(record["round_costs"]) != 3 or record["elite_count"] != 4:
        raise ValueError("unexpected CEM schedule")
    for raw in record["round_costs"]:
        samples = source.cem_candidates(rng, mean, std)
        values = np.asarray(raw)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("invalid CEM costs")
        elite = samples[np.argsort(values, kind="stable")[:4]]
        mean, std = elite.mean(axis=0), elite.std(axis=0)
        pool.append(samples)
        costs.append(values)
    return np.concatenate(pool), np.concatenate(costs)


def main():
    if CANDIDATES.exists() or OUT.exists():
        raise FileExistsError("audit exists")
    plan = source.check_plan()
    result = json.loads(source.RESULT.read_text())
    if result["plan"] != file_hash(source.PLAN) or result["physical"] != file_hash(source.PHYSICAL):
        raise ValueError("result lineage drift")
    parents, probes, tails = source.retention.bank()
    a = source.retention.matrix_chains(parents, probes).reshape(32, 25, 2)
    b = source.retention.matrix_chains(parents, tails)
    arrays, rows = {}, []
    for p, r in zip(plan["rows"], result["rows"], strict=True):
        if p["root"] != r["root"]:
            raise ValueError("root mismatch")
        scores = {}
        common = p["arms"]["uniform"]
        for name in source.ARMS:
            rec = p["arms"][name]
            if rec["logical_rollouts"] != 48 or sum(rec["forward_batches"]) != 48:
                raise ValueError("budget mismatch")
            if name == "cem48":
                candidates, costs = reconstruct_cem(rec)
                arrays[f"r{p['root']}"] = candidates
            else:
                np.testing.assert_array_equal(rec["parent_cost"], common["parent_cost"])
                np.testing.assert_array_equal(rec["probe_cost"], common["probe_cost"])
                kept = source.retention.choose_retained(name, rec["parent_cost"], rec["probe_cost"], 80310 + p["root"])
                if kept != rec["retained"]:
                    raise ValueError("retention mismatch")
                candidates = np.concatenate((parents, a, b[kept].reshape(8, 25, 2)))
                costs = np.concatenate((rec["parent_cost"], np.asarray(rec["probe_cost"]).ravel(), rec["tail_cost"]))
            index = int(np.argmin(costs))
            if index != rec["chosen_index"] or abs(float(costs[index]) - rec["chosen_cost"]) > 1e-10:
                raise ValueError("score selection mismatch")
            np.testing.assert_array_equal(candidates[index], np.asarray(rec["actions"], dtype=np.float32))
            scores[name] = {"index": index, "cost": float(costs[index])}
        rows.append({"root": p["root"], "initial_success": r["initial_success"], "selection": scores})
    with CANDIDATES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with OUT.open("x") as f:
        json.dump({"plan": file_hash(source.PLAN), "result": file_hash(source.RESULT),
                   "runner": file_hash(__file__), "cem_candidates": file_hash(CANDIDATES),
                   "rows": rows, "scope": "selection lineage only, not independent replay"},
                  f, indent=2, allow_nan=False)
    print(f"Validated {len(rows)} roots, 48 scored samples per arm; reconstructed all CEM candidates.")


if __name__ == "__main__":
    main()
