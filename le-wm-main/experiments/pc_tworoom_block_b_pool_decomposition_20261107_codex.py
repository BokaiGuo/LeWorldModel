"""Post-hoc per-task separation of model scoring and description selection."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as source
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(source.PREFIX) + "_b_decomposition.json")


def analyze_task(cost, outcomes, choices):
    cost = np.asarray(cost, dtype=float).reshape(-1)
    if cost.shape != (64,) or not np.isfinite(cost).all() or len(outcomes) != 64:
        raise ValueError("incomplete or nonfinite candidate pool")
    dist = np.asarray([x["final_distance"] for x in outcomes], dtype=float)
    success = np.asarray([x["ever"] for x in outcomes], dtype=bool)
    prefix = np.asarray([x["prefix_success"] for x in outcomes], dtype=bool)
    if not np.isfinite(dist).all():
        raise ValueError("nonfinite physical distance")
    order = np.argsort(cost, kind="stable")
    model = int(order[0])
    best_physical = int(np.argmin(dist))
    ranks = np.empty(64, dtype=int)
    ranks[order] = np.arange(1, 65)
    rho = spearmanr(cost, dist).statistic
    rho = float(rho) if np.isfinite(rho) else None
    row = {
        "pool_success_count": int(success.sum()),
        "pool_tail_success_count": int(np.count_nonzero(success & ~prefix)),
        "model_choice": model,
        "model_success": bool(success[model]),
        "model_prefix_success": bool(prefix[model]),
        "model_distance": float(dist[model]),
        "oracle_best_distance": float(dist[best_physical]),
        "best_success_model_rank": int(ranks[success].min()) if success.any() else None,
        "best_tail_success_model_rank": int(ranks[success & ~prefix].min()) if (success & ~prefix).any() else None,
        "model_vs_distance_spearman": rho,
        "methods": {},
    }
    for method, index in choices.items():
        index = int(index)
        if not 0 <= index < 64:
            raise ValueError("choice out of bounds")
        row["methods"][method] = {
            "index": index, "model_rank": int(ranks[index]),
            "model_cost_regret": float(cost[index] - cost[model]),
            "distance_regret_vs_model": float(dist[index] - dist[model]),
            "success": bool(success[index]), "prefix_success": bool(prefix[index]),
        }
    return row


def proportion(values):
    count, total = sum(values), len(values)
    interval = binomtest(count, total).proportion_ci(confidence_level=0.95, method="wilson")
    return {"count": count, "total": total, "wilson_95": [interval.low, interval.high]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(source.PLAN.read_text())
    physical = json.loads(source.PHYSICAL.read_text())
    original = json.loads(source.RESULT.read_text())
    audit_path = Path(str(source.PREFIX) + "_audit.json")
    audit = json.loads(audit_path.read_text())
    if (plan["pool_sha256"] != file_hash(source.POOL)
            or physical["plan_sha256"] != file_hash(source.PLAN)
            or physical["pool_sha256"] != file_hash(source.POOL)
            or original["physical_sha256"] != file_hash(source.PHYSICAL)
            or audit["physical_sha256"] != file_hash(source.PHYSICAL)):
        raise ValueError("artifact binding mismatch")
    rows = []
    with np.load(source.POOL, allow_pickle=False) as pool:
        if len(plan["rows"]) != 10 or len(physical["rows"]) != 10:
            raise ValueError("panel incomplete")
        for expected, (p, q) in enumerate(zip(plan["rows"], physical["rows"], strict=True)):
            if p["root"] != expected or q["root"] != expected or p["row"] != q["row"]:
                raise ValueError("root ordering mismatch")
            row = analyze_task(pool[f"r{expected}_b_cost"], q["outcomes"], p["choices"])
            row["root"], row["dataset_row"] = expected, p["row"]
            rows.append(row)
    covered = [r for r in rows if r["pool_success_count"]]
    tail_covered = [r for r in rows if r["pool_tail_success_count"]]
    summary = {
        "pool_success": proportion([r["pool_success_count"] > 0 for r in rows]),
        "pool_tail_success": proportion([r["pool_tail_success_count"] > 0 for r in rows]),
        "model_success_all": proportion([r["model_success"] for r in rows]),
        "model_success_given_pool": proportion([r["model_success"] for r in covered]) if covered else None,
        "model_tail_success_given_tail_pool": proportion(
            [r["model_success"] and not r["model_prefix_success"] for r in tail_covered]) if tail_covered else None,
        "model_misses_available_success": sum(r["pool_success_count"] > 0 and not r["model_success"] for r in rows),
        "mean_spearman": float(np.mean([r["model_vs_distance_spearman"] for r in rows
                                       if r["model_vs_distance_spearman"] is not None])),
        "methods": {},
    }
    for method in source.METHODS:
        summary["methods"][method] = {
            "success": proportion([r["methods"][method]["success"] for r in rows]),
            "beats_model_success": sum(r["methods"][method]["success"] and not r["model_success"] for r in rows),
            "loses_model_success": sum(not r["methods"][method]["success"] and r["model_success"] for r in rows),
            "mean_model_cost_regret": float(np.mean([r["methods"][method]["model_cost_regret"] for r in rows])),
            "mean_distance_regret_vs_model": float(np.mean(
                [r["methods"][method]["distance_regret_vs_model"] for r in rows])),
        }
    record = {"rows": rows, "summary": summary, "plan_sha256": file_hash(source.PLAN),
              "pool_sha256": file_hash(source.POOL), "physical_sha256": file_hash(source.PHYSICAL),
              "prior_result_sha256": file_hash(source.RESULT), "audit_sha256": file_hash(audit_path),
              "scope": "post-hoc exposed development, no deployment or gate claim",
              "warning": "Wilson intervals describe binomial sampling only; tasks are not certified independent."}
    with OUTPUT.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
