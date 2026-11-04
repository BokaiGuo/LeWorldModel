"""Rebuild frozen candidates, CEM generations, and all proposed plans."""
import json
from pathlib import Path

from experiments import pc_response_temporal_search_e0_dev_20261104_codex as run

OUTPUT = run.PREFIX.with_name(run.PREFIX.name + "_planning_rebuild.json")


def stable_result(result):
    result = dict(result)
    result["structure"] = {key: value for key, value in result["structure"].items() if key != "svd_seconds"}
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data, audit = json.loads(run.PLANS.read_text()), json.loads(run.AUDIT.read_text())
    if data["dependencies"] != run.dependencies() or audit["plans_sha256"] != run.sha(run.PLANS):
        raise ValueError("planning seal mismatch")
    for index, old in enumerate(data["rows"]):
        fresh = run.planner(old["task"], old["root"], old["arm"])
        if (fresh["trace"] != old["trace"] or stable_result(fresh["result"]) != stable_result(old["result"])
                or any(fresh["counts"][key] != old["counts"][key] for key in
                       ("prediction_calls", "trajectory_samples", "prediction_token_samples", "token_cap"))):
            raise ValueError(f"planning reconstruction mismatch: {old['root']}/{old['arm']}")
        if index % 16 == 15:
            print(f"rebuilt {index + 1}/{len(data['rows'])} plans", flush=True)
    run.write(OUTPUT, {"plans_sha256": run.sha(run.PLANS), "audit_sha256": run.sha(run.AUDIT),
                       "script_sha256": run.sha(Path(__file__)), "reconstructed_plans": len(data["rows"]),
                       "scope": "same-code deterministic reconstruction, not independent replication"})


if __name__ == "__main__":
    main()
