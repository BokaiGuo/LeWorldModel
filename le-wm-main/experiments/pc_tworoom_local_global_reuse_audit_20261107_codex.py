"""Audit source bindings, action identities, and decisions without relabeling outcomes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_local_global_reuse_20261107_codex as exp
from experiments.pc_input_permissions import file_hash

OUT = exp.local.ROOT / "results/pc_tworoom_local_global_reuse_audit_20261107_codex.json"


def audit():
    if OUT.exists():
        raise FileExistsError(OUT)
    exp.global_bank.verify()
    result = json.loads(exp.OUT.read_text())
    local_plan = json.loads(exp.local.PLAN.read_text())
    local_physical = json.loads(exp.local.PHYSICAL.read_text())
    common_plan = json.loads(exp.local.source.PLAN.read_text())
    common_physical = json.loads(exp.local.source.PHYSICAL.read_text())
    full_physical = json.loads(exp.local_full.PHYSICAL.read_text())
    global_result = json.loads(exp.global_bank.RESULT.read_text())
    bindings = {
        "protocol_sha256": exp.PROTOCOL, "runner_sha256": Path(exp.__file__),
        "local_plan_sha256": exp.local.PLAN, "local_pool_sha256": exp.local.POOL,
        "local_physical_sha256": exp.local.PHYSICAL,
        "local_pool_physical_sha256": exp.local_full.PHYSICAL,
        "global_seal_sha256": exp.global_bank.SEAL,
        "global_physical_sha256": exp.global_bank.PHYS,
        "common_physical_sha256": exp.local.source.PHYSICAL,
    }
    if any(result[key] != file_hash(path) for key, path in bindings.items()):
        raise ValueError("result binding mismatch")
    if (global_result["physical_sha256"] != file_hash(exp.global_bank.PHYS)
            or global_result["seal_sha256"] != file_hash(exp.global_bank.SEAL)):
        raise ValueError("global physical evidence mismatch")
    expected = list(range(10, 50))
    for doc in (result, local_plan, local_physical, common_plan, common_physical, full_physical):
        if [r["root"] for r in doc["rows"]] != expected:
            raise ValueError("root order mismatch")
    margins = []
    checked_local = 0
    with (np.load(exp.local.POOL, allow_pickle=False) as local_pool,
          np.load(exp.local.source.POOL, allow_pickle=False) as common_pool,
          np.load(exp.global_bank.SCORES, allow_pickle=False) as global_scores,
          np.load(exp.global_bank.PHYS, allow_pickle=False) as global_physical,
          np.load(exp.global_bank.q.old.POOL, allow_pickle=False) as global_pool):
        for r, plan, phys, prior, prior_phys, pool_phys in zip(
                result["rows"], local_plan["rows"], local_physical["rows"],
                common_plan["rows"], common_physical["rows"], full_physical["rows"], strict=True):
            root = r["root"]
            keep = np.asarray(plan["decisions"]["lookahead"]["retained"])
            wanted = np.sort((keep[:, None] * 8 + np.arange(8)).ravel())
            vals = global_scores[f"r{root}_scores"][wanted]
            margins.append(float(vals.min() - plan["common_cost"]))
            global_id = exp.choose(plan["common_cost"], vals, wanted)
            if global_id != r["global_id"]:
                raise ValueError("global choice differs")
            prior_action = common_pool[f"r{root}_actions"][0]
            if exp.local.source.digest(prior_action) != prior["decisions"]["lookahead"]["action_sha256"]:
                raise ValueError("incumbent digest mismatch")
            actual = local_pool[f"r{root}_actions"][0]
            if exp.local.source.digest(actual) != plan["decisions"]["lookahead"]["action_sha256"]:
                raise ValueError("local selected digest mismatch")
            if r["local_id"] is None and not np.array_equal(actual, prior_action):
                raise ValueError("local incumbent action mismatch")
            records = {x["index"]: x for x in pool_phys["outcomes"]}
            local_actions = local_pool[f"r{root}_local"].reshape(64, 25, 2)
            for idx in wanted:
                record = records[int(idx)]
                if record["action_sha256"] != exp.local.source.digest(local_actions[idx]):
                    raise ValueError("local candidate trace identity mismatch")
                checked_local += 1
            if r["local_id"] is not None:
                if records[r["local_id"]]["trace"] != phys["outcomes"]["lookahead"]["trace"]:
                    raise ValueError("selected physical trace mismatch")
            elif prior_phys["outcomes"]["lookahead"]["trace"] != phys["outcomes"]["lookahead"]["trace"]:
                raise ValueError("incumbent physical trace mismatch")
            if global_id is None:
                expected_outcome = prior_phys["outcomes"]["lookahead"]
                ever, final = expected_outcome["ever"], expected_outcome["final"]
            else:
                action = global_pool[f"r{root}_b"].reshape(64, 25, 2)[global_id]
                if not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
                    raise ValueError("invalid global action")
                term = global_physical[f"r{root}_term"][global_id]
                ever, final = bool(term.any()), bool(term[-1])
            if (ever, final) != (r["global_ever"], r["global_final"]):
                raise ValueError("global physical outcome mismatch")
    summary = {"roots_checked": len(expected), "local_candidate_traces_checked": checked_local,
               "min_global_best_minus_incumbent": min(margins),
               "roots_with_margin_at_most_0_005": sum(x <= 0.005 for x in margins),
               "scope": "same data and simulator, not independent execution"}
    with OUT.open("x") as f:
        json.dump({"result_sha256": file_hash(exp.OUT), "audit_sha256": file_hash(__file__),
                   "summary": summary}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit()
