"""Join model rejection predicates to complete physical capability measurements.

Retrospective diagnosis only: no new selector, threshold, task or P1 admission.
Root identity joins and all 17 programs are mandatory, including all failures.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, verify, TASKS


def indexed(rows):
    result = {}
    for row in rows:
        seed = row["seed"]
        if type(seed) is not int or seed in result:
            raise ValueError("unique explicit integer root identities required")
        result[seed] = row
    return result


def predicates(bank, decision):
    if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
        raise ValueError("complete ordered original bank required")
    greedy = min(range(17), key=lambda i: (bank[i]["prefix_cost"], i))
    if decision["greedy_index"] != greedy:
        raise ValueError("original greedy identity disagrees with model costs")
    for b in bank:
        gain = b["common_response_gain"]
        numbers = [b["prefix_cost"], b["variants"]["cem_then_gn4"]["terminal_cost"],
                   gain["bounded_quadratic_gain"], gain["bounded_quadratic_gain_upper_bound"]]
        if (any(not math.isfinite(float(v)) for v in numbers)
                or gain.get("objective_certificate_passed") is not True
                or gain.get("physical_labels_used") is not False
                or numbers[2] > numbers[3]):
            raise ValueError("finite certified model-only comparisons required")
    threshold = max(bank[i]["common_response_gain"]["bounded_quadratic_gain_upper_bound"]
                    for i in (0, greedy))
    rows = []
    for b in bank:
        i = b["index"]
        checks = {
            "nonzero_program": i != 0,
            "model_stage_worse_than_zero": b["prefix_cost"] > bank[0]["prefix_cost"] + 1e-6,
            "model_terminal_better_than_greedy": b["variants"]["cem_then_gn4"]["terminal_cost"]
                < bank[greedy]["variants"]["cem_then_gn4"]["terminal_cost"] - 1e-6,
            "model_common_gain_certifiably_better_than_controls":
                b["common_response_gain"]["bounded_quadratic_gain"] > threshold,
        }
        rows.append({"index": i, "checks": checks,
                     "rejected_by": [k for k, v in checks.items() if not v],
                     "qualified": all(checks.values()),
                     "common_gain_minus_control_upper":
                         b["common_response_gain"]["bounded_quadratic_gain"] - threshold})
    qualified = [r["index"] for r in rows if r["qualified"]]
    native = min(qualified, key=lambda i: (bank[i]["variants"]["cem_then_gn4"]["terminal_cost"], i)) if qualified else greedy
    if qualified != decision["qualified_indices"] or native != decision["native_index"]:
        raise ValueError("independent predicate reconstruction changed original selector")
    return rows


def audit(capability_path, reference_path):
    # Independent analyzer verifies complete archived CUDA records, all probes,
    # original prefix replay and exact alignment to the earlier partial audit.
    capability = verify(capability_path)
    if not capability["all_programs"]:
        raise ValueError("all17 physical coverage required; partial roles cannot establish bank absence")
    raw = json.loads(capability_path.read_text())
    bindings = raw["sealed_protocol"]["input_bindings"]
    predictions_path = reference_path.with_suffix(".predictions.json")
    for p in (reference_path, predictions_path):
        if bindings.get(str(p.resolve())) != digest(p):
            raise ValueError("same original bound model/physical source required")
    predictions = indexed(json.loads(predictions_path.read_text())["rows"])
    measured = indexed(capability["rows"])
    if set(predictions) != set(measured):
        raise ValueError("duplicate-free root set equality required")
    rows = []
    for seed in sorted(measured):
        tasks = {}
        for task in TASKS:
            plan = predictions[seed]["tasks"][task]
            actual = measured[seed]["tasks"][task]
            records = predicates(plan["bank"], plan["decisions"]["cem_then_gn4"])
            for r in records:
                i = r["index"]
                diagnosis = actual["mechanism_diagnostics"][f"program_{i}"]
                r["physical_checks"] = diagnosis["checks"]
                r["all_physical_diagnostic_checks"] = diagnosis["all_diagnostic_checks"]
                for space, label in (("physical_rgb_fd", "physical_rgb"), ("model_fd", "actual_history_model_fd")):
                    differences = [[actual["cells"][str(i)]["probes"][k]["fits"][space]["common"]["gain_fraction"]
                        - actual["cells"][str(actual["roles"][control])]["probes"][k]["fits"][space]["common"]["gain_fraction"]
                        for control in ("zero", "greedy")] for k in range(2)]
                    r[label + "_common_gain_differences"] = differences
                    r[label + "_gain_better_at_both_scales"] = all(v > 1e-10 for scale in differences for v in scale)
                r["diagnostic_disagreements"] = []
                if r["all_physical_diagnostic_checks"]:
                    if not r["checks"]["model_stage_worse_than_zero"]:
                        r["diagnostic_disagreements"].append("model_vs_physical_stage_sign")
                    if not r["physical_rgb_gain_better_at_both_scales"]:
                        r["diagnostic_disagreements"].append("physical_joint_vs_rgb_common_gain")
                    elif not r["actual_history_model_fd_gain_better_at_both_scales"]:
                        r["diagnostic_disagreements"].append("actual_history_model_vs_physical_rgb_common_gain")
                    if (r["actual_history_model_fd_gain_better_at_both_scales"] and not
                            r["checks"]["model_common_gain_certifiably_better_than_controls"]):
                        r["diagnostic_disagreements"].append("old_predicted_state_or_reference_residual_vs_actual_history_fd")
                r["code_defect_confirmed"] = False  # Needs a causal correction, not this retrospective join.
            tasks[task] = {"programs": records, "original_decision": plan["decisions"]["cem_then_gn4"]}
        rows.append({"seed": seed, "tasks": tasks})
    aggregate = {}
    for task in TASKS:
        records = [p for row in rows for p in row["tasks"][task]["programs"]]
        candidates = [p for p in records if p["all_physical_diagnostic_checks"]]
        aggregate[task] = {
            "complete_program_cells": len(records),
            "complete_roots": len(rows),
            "roots_with_physical_diagnostic_candidate": sum(any(p["all_physical_diagnostic_checks"]
                for p in row["tasks"][task]["programs"]) for row in rows),
            "physical_diagnostic_candidates": len(candidates),
            "physical_candidates_rejected_by_model": sum(not p["qualified"] for p in candidates),
            "physical_candidates_rejected_only_by_common_gain": sum(p["rejected_by"] ==
                ["model_common_gain_certifiably_better_than_controls"] for p in candidates),
            "physical_joint_candidates_without_rgb_common_gain": sum(not
                p["physical_rgb_gain_better_at_both_scales"] for p in candidates),
            "physical_candidates_actual_model_fd_gain_recognizes": sum(
                p["actual_history_model_fd_gain_better_at_both_scales"] for p in candidates),
            "diagnostic_disagreements_nonexclusive": {key: sum(key in p["diagnostic_disagreements"] for p in candidates)
                for key in sorted({k for p in candidates for k in p["diagnostic_disagreements"]})},
            "candidate_rejection_reasons": {key: sum(key in p["rejected_by"] for p in candidates)
                for key in records[0]["checks"]},
            "model_qualified_without_physical_conjunction": sum(p["qualified"] and not p["all_physical_diagnostic_checks"] for p in records),
        }
    return {"schema": "reacher_d2p_capability_rejections.v1", "rows": rows, "aggregate": aggregate,
            "bindings": {str(p.resolve()): digest(p) for p in (capability_path, reference_path, predictions_path)},
            "implementation_sha256": digest(Path(__file__)), "P1_admission": False,
            "physical_labels_used_for_deployment": False,
            "scope": "Known-eight-root all17 retrospective attribution, not independent efficacy, new admission or proof of a code defect. Local finite-difference geometry is not global controllability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable diagnostic output required")
    result = audit(args.capability, args.reference)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
