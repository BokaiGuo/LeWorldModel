"""Summarize every sealed physical confirmation root, including abstentions.

This is a diagnostic report, not a formal P1 calibration, outcome, or budget
admission. No thresholds or model parameters are chosen here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(report: dict) -> dict:
    if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("completed, integrity-passing physical report required")
    protocol = report["sealed_protocol"]
    rows = report["rows"]
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(protocol["root_seeds"]):
        raise ValueError("missing or duplicate root identity")
    by_seed = {row["seed"]: row for row in rows}
    cells = []
    for seed in sorted(by_seed):
        root = by_seed[seed]
        selected = root["reconfiguration"]
        eligible = selected is not None
        if bool(root["model_selection"]["non_greedy_witness_found"]) != eligible:
            raise ValueError("model eligibility changed after physical execution")
        zero, greedy = root["zero_prefix"], root["same_bank_greedy"]
        policy = selected if eligible else zero
        final = float(policy["physical"]["final_qpos_distance"])
        zero_final = float(zero["physical"]["final_qpos_distance"])
        greedy_final = float(greedy["physical"]["final_qpos_distance"])
        if not np.isfinite([final, zero_final, greedy_final]).all():
            raise ValueError("nonfinite physical distance")
        cell = {"seed": seed, "source_episode": root["source_episode"], "model_eligible": eligible,
            "executed_policy": "selected_reconfiguration" if eligible else "executed_zero_prefix_fallback",
            "final_qpos_distance": final, "zero_final_qpos_distance": zero_final,
            "greedy_final_qpos_distance": greedy_final,
            "final_minus_greedy": final - greedy_final, "final_minus_zero": final - zero_final}
        if eligible:
            observed = float(selected["observed_first_chunk"]["candidate_goal_cost"])
            observed_zero = float(zero["observed_first_chunk"]["candidate_goal_cost"])
            observed_greedy = float(greedy["observed_first_chunk"]["candidate_goal_cost"])
            qpos = float(selected["physical"]["after_first_chunk_qpos_distance"])
            qpos_zero = float(zero["physical"]["after_first_chunk_qpos_distance"])
            qpos_greedy = float(greedy["physical"]["after_first_chunk_qpos_distance"])
            cell.update({"rgb_non_greedy_vs_zero": observed >= observed_zero - 1e-8,
                "rgb_non_greedy_vs_same_bank_greedy": observed >= observed_greedy - 1e-8,
                "qpos_non_greedy_vs_zero": qpos >= qpos_zero - 1e-8,
                "qpos_non_greedy_vs_same_bank_greedy": qpos >= qpos_greedy - 1e-8,
                "future_model_gain_over_same_bank_greedy": root["model_selection"]["future_reference_score_improvement"]})
        cells.append(cell)
    eligible = [row for row in cells if row["model_eligible"]]
    return {"schema": "reacher_d2p_backtracking_confirmation_summary.v1", "status": "completed",
        "all_root_count": len(cells), "eligible_root_count": len(eligible),
        "abstention_root_count": len(cells) - len(eligible),
        "eligible_mechanism_counts": {key: sum(row[key] for row in eligible) for key in (
            "rgb_non_greedy_vs_zero", "rgb_non_greedy_vs_same_bank_greedy",
            "qpos_non_greedy_vs_zero", "qpos_non_greedy_vs_same_bank_greedy")},
        "eligible_final_better_than_both_controls": sum(row["final_minus_greedy"] < 0 and row["final_minus_zero"] < 0 for row in eligible),
        "all_root_neutral_fallback_diagnostic": {
            "mean_final_minus_greedy": float(np.mean([row["final_minus_greedy"] for row in cells])),
            "mean_final_minus_zero": float(np.mean([row["final_minus_zero"] for row in cells])),
            "wins_against_greedy": sum(row["final_minus_greedy"] < 0 for row in cells),
            "losses_against_greedy": sum(row["final_minus_greedy"] > 0 for row in cells)},
        "rows": cells,
        "claim_boundary": "All-root exploratory physical diagnostic including actual neutral fallbacks. Not formal P1, held-out outcome efficacy, a compute-matched comparison, or proof of a controllability-gap family."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(json.loads(args.report.read_text()))
    result["inputs"] = {"report": str(args.report.resolve()), "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
                       "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "inputs")}))


if __name__ == "__main__":
    main()
