"""Retrospective equal-query comparison; no physical labels in selection."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_conditional_local_tail_20261107_codex as local
from experiments import pc_tworoom_full_b_pool_diagnostic_20261107_codex as global_bank
from experiments import pc_conditional_local_tail_full_pool_20261107_codex as local_full
from experiments.pc_input_permissions import file_hash

OUT = local.ROOT / "results/pc_tworoom_local_global_reuse_20261107_codex.json"
PROTOCOL = local.ROOT.parent / "docs/PC_TWOROOM_LOCAL_VS_GLOBAL_REUSE_PROTOCOL_20261107_CODEX.md"


def choose(incumbent_cost, costs, ids):
    costs = np.asarray(costs, dtype=float)
    ids = np.asarray(ids)
    if (costs.shape != (16,) or ids.shape != (16,) or not np.isfinite(costs).all()
            or not np.isfinite(incumbent_cost) or len(np.unique(ids)) != 16):
        raise ValueError("expected 16 unique, finite scores")
    index = int(np.argmin(costs))
    return int(ids[index]) if costs[index] < incumbent_cost else None


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    global_bank.verify()
    plan = json.loads(local.PLAN.read_text())
    phys = json.loads(local.PHYSICAL.read_text())
    common_phys = json.loads(local.source.PHYSICAL.read_text())
    pool_phys = json.loads(local_full.PHYSICAL.read_text())
    if (plan["pool_sha256"] != file_hash(local.POOL)
            or phys["plan_sha256"] != file_hash(local.PLAN)
            or phys["pool_sha256"] != file_hash(local.POOL)):
        raise ValueError("local binding mismatch")
    if (common_phys["plan_sha256"] != file_hash(local.source.PLAN)
            or common_phys["pool_sha256"] != file_hash(local.source.POOL)
            or pool_phys["source_plan_sha256"] != file_hash(local.PLAN)
            or pool_phys["source_pool_sha256"] != file_hash(local.POOL)
            or pool_phys["source_physical_sha256"] != file_hash(local.PHYSICAL)):
        raise ValueError("physical source binding mismatch")
    if [r["root"] for r in plan["rows"]] != list(range(10, 50)):
        raise ValueError("wrong panel")
    physical_rows = {r["root"]: r for r in phys["rows"]}
    common_rows = {r["root"]: r for r in common_phys["rows"]}
    pool_rows = {r["root"]: r for r in pool_phys["rows"]}
    rows = []
    with (np.load(local.POOL, allow_pickle=False) as lp,
          np.load(global_bank.SCORES, allow_pickle=False) as gs,
          np.load(global_bank.PHYS, allow_pickle=False) as gp,
          np.load(global_bank.q.old.POOL, allow_pickle=False) as actions):
        for row in plan["rows"]:
            root = row["root"]
            keep = np.asarray(row["decisions"]["lookahead"]["retained"])
            wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
            ids, scores = lp[f"r{root}_ids"], lp[f"r{root}_scores"]
            positions = np.searchsorted(ids, wanted)
            if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], wanted):
                raise ValueError("missing local candidate")
            decision = row["decisions"]["lookahead"]
            local_id = choose(row["common_cost"], scores[positions], wanted)
            if (decision["bank"], decision["index"]) != (
                    ("common", -1) if local_id is None else ("local", local_id)):
                raise ValueError("local selection mismatch")
            nonlocal_id = choose(row["common_cost"], gs[f"r{root}_scores"][wanted], wanted)
            term = gp[f"r{root}_term"]
            if term.shape != (64, 25) or term.dtype != np.bool_:
                raise ValueError("invalid physical bank")
            bank = actions[f"r{root}_b"].reshape(64, 25, 2)
            if not np.isfinite(bank).all() or np.any(np.abs(bank) > 1):
                raise ValueError("invalid action")
            local_actions = lp[f"r{root}_local"].reshape(64, 25, 2)
            if not np.array_equal(bank[wanted, :10], local_actions[wanted, :10]):
                raise ValueError("prefix mismatch")
            prior = physical_rows[root]["outcomes"]["lookahead"]
            # If the local winner is not common, its outcome cannot be reused
            # for a nonlocal incumbent selection; recover the common outcome
            # from the preceding experiment, where all arms share the chain.
            if nonlocal_id is None:
                common = common_rows[root]["outcomes"]["lookahead"]
                outcome = {"ever": common["ever"], "final": common["final"]}
            else:
                outcome = {"ever": bool(term[nonlocal_id].any()), "final": bool(term[nonlocal_id, -1])}
            local_outcomes = {x["index"]: x for x in pool_rows[root]["outcomes"]}
            if any(int(i) not in local_outcomes for i in wanted):
                raise ValueError("missing local physical outcome")
            local_pool_success = any(any(t["terminated"] for t in local_outcomes[int(i)]["trace"])
                                     for i in wanted)
            rows.append({"root": root, "local_id": local_id, "global_id": nonlocal_id,
                         "local_ever": prior["ever"], "local_final": prior["final"],
                         "global_ever": outcome["ever"], "global_final": outcome["final"],
                         "local_pool_has_success": local_pool_success,
                         "global_pool_has_success": bool(term[wanted].any())})
    summary = {}
    for measure in ("ever", "final"):
        a = [r[f"local_{measure}"] for r in rows]
        b = [r[f"global_{measure}"] for r in rows]
        summary[measure] = {"local": sum(a), "global": sum(b),
                            "global_only": sum(y and not x for x, y in zip(a, b)),
                            "local_only": sum(x and not y for x, y in zip(a, b))}
    summary["global_candidate_selected"] = sum(r["global_id"] is not None for r in rows)
    summary["local_pool_has_success"] = sum(r["local_pool_has_success"] for r in rows)
    summary["global_pool_has_success"] = sum(r["global_pool_has_success"] for r in rows)
    summary["development_exit"] = (summary["ever"]["global_only"] >= 2
                                    and summary["ever"]["local_only"] <= 1)
    payload = {"scope": "retrospective, exposed development; not independent",
               "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
               "local_plan_sha256": file_hash(local.PLAN), "local_pool_sha256": file_hash(local.POOL),
               "local_physical_sha256": file_hash(local.PHYSICAL),
               "local_pool_physical_sha256": file_hash(local_full.PHYSICAL),
               "global_seal_sha256": file_hash(global_bank.SEAL),
               "global_physical_sha256": file_hash(global_bank.PHYS),
               "common_physical_sha256": file_hash(local.source.PHYSICAL),
               "summary": summary, "rows": rows}
    with OUT.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
