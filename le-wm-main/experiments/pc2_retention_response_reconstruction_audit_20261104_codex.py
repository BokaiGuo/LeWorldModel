"""Reconstruct the saved PC2 retention intervention conclusions.

This audit does not run a model or physics.  It verifies source-plan binding,
recomputes control identity and all saved comparison/qualification booleans,
and records that the saved ``lookahead`` arm is ridge-value selection rather
than ordinary suffix lookahead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import pc_retention_equal_slot_cost_dev_20261104_codex as source
from experiments import pc2_retention_response_same_chain_mechanism_20261104_codex as mechanism


OUT = source.ROOT / "results/pc2_retention_response_same_chain_mechanism_20261104_codex"
REPORT = OUT / "reconstructed_audit_20261104.json"
CONTROLS = ("selected", "zero", "greedy", "lookahead", "random")
SCALES = ("0.01", "0.05")


def _recompute_comparisons(item: dict) -> dict:
    suffix = item["suffix"]
    measurements = item["measurements"]
    selected = suffix["selected"]
    comparisons = {}
    for name in CONTROLS[1:]:
        other = suffix[name]
        common_gain = bool(
            selected["relative_zero"][-1]
            < other["relative_zero"][-1] - 1e-10
        )
        replan_gain = bool(
            np.mean([x["score"] for x in selected["runs"]])
            < np.mean([x["score"] for x in other["runs"]]) - 1e-10
        )
        gaps = {}
        for epsilon in SCALES:
            left, right = measurements["selected"][epsilon], measurements[name][epsilon]
            matrix = float(
                np.linalg.norm(np.asarray(left["matrix"]) - np.asarray(right["matrix"]))
            )
            lg, rg = left["goal_gradient"], right["goal_gradient"]
            gradient = None if lg is None or rg is None else float(
                np.linalg.norm(np.asarray(lg) - np.asarray(rg))
            )
            gaps[epsilon] = {"matrix": matrix, "goal_gradient": gradient}
        response_diff = all(
            value["matrix"] > 1e-6
            and value["goal_gradient"] is not None
            and value["goal_gradient"] > 1e-6
            for value in gaps.values()
        )
        branches_differ = (
            measurements["selected"]["0.01"]["branches"][0]
            != measurements[name]["0.01"]["branches"][0]
        )
        comparisons[name] = {
            "common_gain": common_gain,
            "replan_gain": replan_gain,
            "response_diff": response_diff,
            "gaps": gaps,
            "nominal_branch_diff": branches_differ,
        }
    return comparisons


def _recompute_qualification(item: dict, comparisons: dict) -> bool:
    chain = np.asarray(item["chain"], np.float32)
    nonzero = bool(np.any(chain[:5] != 0) and np.any(chain[5:] != 0))
    nonprogress = bool(item["nonprogress"])
    stable = all(
        item["measurements"][name][epsilon]["all_branch_stable"]
        for name in CONTROLS
        for epsilon in SCALES
    )
    return bool(
        nonzero
        and nonprogress
        and all(
            comparisons[name]["common_gain"]
            and comparisons[name]["replan_gain"]
            and comparisons[name]["response_diff"]
            for name in CONTROLS[1:]
        )
        and stable
    )


def audit() -> dict:
    rows = []
    failures = []
    for root, task in enumerate(source.tasks()):
        path = OUT / f"root{root:02d}.json"
        item = json.loads(path.read_text())
        source_plan_path = source.OUT / f"plan_{root:02d}_response.json"
        source_plan = json.loads(source_plan_path.read_text())
        saved_chain = np.asarray(item["chain"], np.float32)
        source_chain = np.asarray(source_plan["result"]["selected"], np.float32)
        if not np.array_equal(saved_chain, source_chain):
            failures.append({"root": root, "reason": "selected_chain_not_bound_to_source_plan"})
            continue
        layer = source_plan["result"]["child"]["layers"][0]
        greedy = int(np.argmin([candidate["score"] for candidate in layer["candidates"]]))
        ridge = min(
            layer["reachability"],
            key=lambda candidate: (candidate["value"], candidate["candidate"]),
        )["candidate"]
        controls = item["controls"]
        if not np.array_equal(
            np.asarray(controls["greedy"], np.float32),
            np.asarray(layer["candidates"][greedy]["chain"], np.float32),
        ):
            failures.append({"root": root, "reason": "greedy_control_mismatch"})
            continue
        if not np.array_equal(
            np.asarray(controls["lookahead"], np.float32),
            np.asarray(layer["candidates"][ridge]["chain"], np.float32),
        ):
            failures.append({"root": root, "reason": "ridge_control_mismatch"})
            continue
        rebuilt = _recompute_comparisons(item)
        qualified = _recompute_qualification(item, rebuilt)
        if rebuilt != item["comparisons"]:
            failures.append({"root": root, "reason": "comparisons_not_reproducible"})
        if qualified != bool(item["smooth_mechanism_qualified"]):
            failures.append({"root": root, "reason": "qualification_not_reproducible"})
        rows.append(
            {
                "root": root,
                "status": item["status"],
                "source_plan_sha256": item["source_plan_sha256"],
                "selected_chain_bound": True,
                "greedy_index": greedy,
                "ridge_value_index": ridge,
                "lookahead_definition": "local_ridge_reachability_value_minimum",
                "ordinary_suffix_lookahead_present": False,
                "comparisons_recomputed": rebuilt,
                "smooth_mechanism_qualified_recomputed": qualified,
                "saved_qualified": bool(item["smooth_mechanism_qualified"]),
                "final_distance": float(item["suffix"]["selected"]["runs"][0]["score"])
                if item["suffix"]["selected"]["runs"]
                else None,
            }
        )
    return {
        "schema": "pc2_retention_response_reconstructed_audit.v1",
        "status": "passed" if not failures else "failed",
        "rows": rows,
        "failures": failures,
        "ordinary_suffix_lookahead_present": False,
        "lookahead_control_name": "ridge_value_optimal_prefix",
        "scope": "saved same-chain development records; no inference or physics",
        "gate_status": {"PC2": "not_passed", "reason": "development reconstruction only"},
    }


if __name__ == "__main__":
    if REPORT.exists():
        raise FileExistsError(REPORT)
    result = audit()
    mechanism.journal.write_json(REPORT, result)
    print(json.dumps(
        {
            "status": result["status"],
            "rows": len(result["rows"]),
            "failures": result["failures"],
            "ordinary_suffix_lookahead_present": result[
                "ordinary_suffix_lookahead_present"
            ],
        },
        indent=2,
    ))
