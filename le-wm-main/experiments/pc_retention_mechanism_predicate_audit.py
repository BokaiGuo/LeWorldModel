"""Independently recompute predeclared smooth-mechanism booleans."""
import json
import numpy as np

from experiments import pc_retention_mechanism_next_dev as source


def main():
    out = source.OUT
    target = out / "predicate_audit.json"
    if target.exists():
        raise FileExistsError(target)
    plans = json.loads((out / "plans.json").read_text())
    data = json.loads((out / "interventions.json").read_text())
    summaries = []
    for plan, row in zip(plans["rows"], data["rows"], strict=True):
        if row["status"] != "completed":
            summaries.append({"root": row["root"], "arm": row["arm"], "status": row["status"]})
            continue
        chain = np.asarray(plan["result"]["selected"], np.float32)
        nonzero = bool(np.any(chain[:5] != 0) and np.any(chain[5:] != 0))
        if nonzero != row["nonzero"]:
            raise ValueError("nonzero mismatch")
        for name, saved in row["comparisons"].items():
            selected, other = row["suffix"]["selected"], row["suffix"][name]
            common = selected["relative_zero"][-1] < other["relative_zero"][-1] - 1e-10
            replan = np.mean([x["score"] for x in selected["runs"]]) < np.mean(
                [x["score"] for x in other["runs"]]) - 1e-10
            valid = []
            for eps in ("0.01", "0.05"):
                left, right = row["measurements"]["selected"][eps], row["measurements"][name][eps]
                matrix = float(np.linalg.norm(np.asarray(left["matrix"]) - right["matrix"]))
                lg, rg = left["goal_gradient"], right["goal_gradient"]
                gradient = None if lg is None or rg is None else float(np.linalg.norm(np.asarray(lg) - rg))
                if {"matrix": matrix, "goal_gradient": gradient} != saved["gaps"][eps]:
                    raise ValueError("response gap mismatch")
                valid.append(matrix > 1e-6 and gradient is not None and gradient > 1e-6)
            nominal_branch_diff = (row["measurements"]["selected"]["0.01"]["branches"][0] !=
                                   row["measurements"][name]["0.01"]["branches"][0])
            if (bool(common), bool(replan), all(valid), nominal_branch_diff) != (
                    saved["common_gain"], saved["replan_gain"], saved["response_diff"], saved["nominal_branch_diff"]):
                raise ValueError("comparison mismatch")
        def qualifies(names):
            return bool(nonzero and row["nonprogress"] and
                all(all(row["comparisons"][name][field] for field in ("common_gain", "replan_gain", "response_diff"))
                    for name in names) and all(row["measurements"][name][eps]["all_branch_stable"]
                    for name in ("selected", *names) for eps in ("0.01", "0.05")))
        base, full = qualifies(("zero", "greedy")), qualifies(("zero", "greedy", "lookahead", "random"))
        if base != row["base_intersection"] or full != row["all_intersection"]:
            raise ValueError("intersection mismatch")
        summaries.append({"root": row["root"], "arm": row["arm"], "base": base, "all": full})
    source.journal.write_json(target, {"status": "passed", "rows": summaries,
        "interventions_sha256": source.seal.digest(out / "interventions.json"),
        "scope": "same-environment numerical predicate reconstruction"})
    print(f"Verified {len(summaries)} predeclared mechanism predicate records.", flush=True)


if __name__ == "__main__":
    main()
