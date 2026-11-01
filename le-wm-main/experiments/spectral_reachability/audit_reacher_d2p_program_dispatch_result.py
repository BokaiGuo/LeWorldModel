"""Audit all-root dispatch effects, exact fallbacks, and measured compute.

Neutral policy comparison joins physically executed branches by root identity;
it is a policy trace counterfactual, not another independent experimental arm.
No result here admits formal P1 or changes the frozen runtime selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import reacher_d2p_program_dispatch as dispatcher
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import root_execution_gates


def summarize(report: dict) -> dict:
    if report.get("schema") != "reacher_d2p_program_dispatch_diagnostic.v1" or report.get("status") != "completed":
        raise ValueError("completed dispatch diagnostic required")
    if not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("integrity gates failed")
    roots, protocol = report["rows"], report["sealed_protocol"]
    seeds = [row["seed"] for row in roots]
    if not seeds or len(set(seeds)) != len(seeds) or sorted(seeds) != sorted(protocol["root_seeds"]):
        raise ValueError("all sealed root identities required exactly once")
    if protocol["policy"] != dispatcher.POLICY:
        raise ValueError("runtime policy differs")
    cells, compute = [], {name: [] for name in ("dispatch", "same_bank_greedy", "zero", "lookahead_cem")}
    mechanisms = {key: 0 for key in ("rgb_non_greedy_vs_zero", "rgb_non_greedy_vs_greedy", "qpos_non_greedy_vs_zero", "qpos_non_greedy_vs_greedy")}
    for root in sorted(roots, key=lambda item: item["seed"]):
        methods = root["methods"]
        if not all(root_execution_gates(methods).values()):
            raise ValueError("live branch identity/replay contract failed")
        receipt = methods["dispatch"]["first_receipt"]
        decision = dispatcher.dispatch_decision(receipt["reconfiguration_compilation"])
        if decision != receipt["dispatch"]:
            raise ValueError("stored dispatch differs from model-only recomputation")
        eligible = decision["reconfiguration_witness_found"]
        distances = {name: float(method["physical"]["final_qpos_distance"]) for name, method in methods.items()}
        neutral = distances["dispatch"] if eligible else distances["zero"]
        cells.append({"seed": root["seed"], "source_episode": root["source_episode"], "model_eligible": eligible,
            "dispatch_minus_greedy": distances["dispatch"] - distances["same_bank_greedy"],
            "dispatch_minus_zero": distances["dispatch"] - distances["zero"],
            "dispatch_minus_cem": distances["dispatch"] - distances["lookahead_cem"],
            "greedy_minus_cem": distances["same_bank_greedy"] - distances["lookahead_cem"],
            "neutral_minus_greedy": neutral - distances["same_bank_greedy"],
            "dispatch_minus_neutral": distances["dispatch"] - neutral})
        for name, method in methods.items():
            actual = method["actual_compute"]
            suffix = method["suffix_receipt"]
            first = method["first_receipt"]
            expected_first = {"dispatch": receipt["model_predict_sample_evaluations"], "same_bank_greedy": protocol["proposals"],
                "zero": 0, "lookahead_cem": protocol["cem"]["samples"] * protocol["cem"]["steps"] * protocol["cem"]["horizon"]}[name]
            if actual["model_predict_sample_evaluations"] != expected_first + suffix["model_predict_sample_evaluations"]:
                raise ValueError("measured prediction samples differ from declared work")
            if actual["logical_backward_vjp_rows"] != first.get("backward_vjp_rows", 0) + suffix["backward_vjp_rows"]:
                raise ValueError("logical backward work differs from declared work")
            compute[name].append({**actual, "planner_wall_seconds": method["planner_wall_seconds"]})
        if eligible:
            candidate, zero, greedy = (methods[name] for name in ("dispatch", "zero", "same_bank_greedy"))
            rgb = lambda branch: branch["observed_first_chunk"]["candidate_goal_cost"]
            qpos = lambda branch: branch["physical"]["after_first_chunk_qpos_distance"]
            mechanisms["rgb_non_greedy_vs_zero"] += rgb(candidate) >= rgb(zero) - 1e-8
            mechanisms["rgb_non_greedy_vs_greedy"] += rgb(candidate) >= rgb(greedy) - 1e-8
            mechanisms["qpos_non_greedy_vs_zero"] += qpos(candidate) >= qpos(zero) - 1e-8
            mechanisms["qpos_non_greedy_vs_greedy"] += qpos(candidate) >= qpos(greedy) - 1e-8
    paired = {}
    for key in ("dispatch_minus_greedy", "dispatch_minus_zero", "dispatch_minus_cem", "greedy_minus_cem", "neutral_minus_greedy", "dispatch_minus_neutral"):
        values = np.asarray([row[key] for row in cells])
        paired[key] = {"n": len(cells), "mean": float(values.mean()),
            "ci95_descriptive_root_bootstrap": p0._bootstrap_mean(values, int(min(seeds)) + 901),
            "wins": int((values < 0).sum()), "losses": int((values > 0).sum()), "ties": int((values == 0).sum())}
    summary_compute = {name: {key: {"min": float(min(row[key] for row in values)),
        "max": float(max(row[key] for row in values)), "mean": float(np.mean([row[key] for row in values]))}
        for key in ("model_predict_calls", "model_predict_sample_evaluations", "logical_backward_vjp_rows", "planner_wall_seconds")}
        for name, values in compute.items()}
    return {"schema": "reacher_d2p_program_dispatch_result_audit.v1", "status": "completed",
        "root_count": len(cells), "reconfiguration_count": sum(row["model_eligible"] for row in cells),
        "progress_fallback_count": sum(not row["model_eligible"] for row in cells),
        "mechanism_counts_on_eligible": {key: int(value) for key, value in mechanisms.items()},
        "paired": paired, "compute": summary_compute, "rows": cells,
        "claim_boundary": "All-root fresh diagnostic and physical-branch policy replay; no threshold fitting, formal P1 admission, held-out calibration/outcome efficacy, full resource matching, or theory verdict."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()))
    result["inputs"] = {"report": str(args.report.resolve()), "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "inputs", "compute")}, sort_keys=True))


if __name__ == "__main__":
    main()
