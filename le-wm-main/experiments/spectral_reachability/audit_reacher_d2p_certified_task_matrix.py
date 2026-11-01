"""Identity-joined scalar audit of all three fixed certified task conditions.

No neural model, simulator or physical label enters a new decision here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_actuator_limit_tasks import TASKS, action_limit_gate
from experiments.spectral_reachability import reacher_d2p_certified_geometry_dispatch as compiler
from experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration import _paired


def audit(reports):
    if set(reports) != set(TASKS):
        raise ValueError("all three predeclared tasks required, no task selection")
    reference = reports["full_drive"]["sealed_protocol"]
    lookup, aggregates = {}, {}
    for task, report in reports.items():
        protocol = report["sealed_protocol"]
        if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
            raise ValueError("every task must have completed integrity-passing evidence")
        if protocol["compiler_variant"] != "geometry_certified" or protocol["task"] != task or list(protocol["raw_actuator_limits"]) != list(TASKS[task]):
            raise ValueError("certified compiler and exact fixed torque contract required")
        if any(protocol[key] != reference[key] for key in ("root_seeds", "source_roots", "weights_sha256", "adapter_sha256", "source_sha256", "target_delta", "implementation_sha256")):
            raise ValueError("matched source roots, goal, model and implementation required across tasks")
        rows = {row["seed"]: row for row in report["rows"]}
        if len(rows) != len(report["rows"]) or set(rows) != set(reference["root_seeds"]):
            raise ValueError("duplicate-free complete root identity sets required")
        lookup[task] = rows
        for seed, row in rows.items():
            if row["task"] != task or not all(row["gates"].values()) or row["fd_preflight"]["median_relative_error"] > .1:
                raise ValueError("every root must satisfy declared task FD and execution contracts")
            if any(not action_limit_gate(method, TASKS[task]) for method in row["methods"].values()):
                raise ValueError("future actions exceed predeclared task limits")
            receipt = row["compilation"]
            geometry = {int(key): value for key, value in receipt["common_task_response"]["candidate_metrics"].items()}
            decision = compiler.qp.qualify_certified_decision(receipt, geometry)
            decision["policy"] = compiler.POLICY
            if decision != receipt["dispatch"]:
                raise ValueError("stored selector decision fails certified model-only replay")
            if not all(cell["qp_certificate"]["objective_duality_gap_upper_bound"] <= 1e-8 * max(1, cell["baseline_squared_residual"])
                       for cell in geometry.values()):
                raise ValueError("objective error certificate exceeds frozen tolerance")
        ordered = [rows[seed] for seed in sorted(rows)]
        eligible = [row for row in ordered if row["compilation"]["dispatch"]["reconfiguration_witness_found"]]
        aggregates[task] = {"roots": len(ordered), "model_witnesses": len(eligible),
            "witness_actual_rgb_non_greedy": sum(row["mechanism"]["actual_rgb_non_greedy_vs_zero"] for row in eligible),
            "witness_actual_qpos_non_greedy": sum(row["mechanism"]["actual_qpos_non_greedy_vs_zero"] for row in eligible),
            "mean_final_distance": {name: float(np.mean([row["methods"][name]["physical"]["final_qpos_distance"] for row in ordered]))
                for name in ("dispatch", "same_bank_greedy", "zero", "cem_mpc")},
            "paired": {name: _paired(ordered, "dispatch", name, reference["bootstrap_seed"]) for name in ("same_bank_greedy", "zero", "cem_mpc")},
            "compute_means": {name: {metric: float(np.mean([row["methods"][name]["actual_compute"][metric] for row in ordered]))
                for metric in ("model_predict_sample_evaluations", "logical_backward_vjp_rows")}
                for name in ("dispatch", "same_bank_greedy", "zero", "cem_mpc")}}
    for seed in reference["root_seeds"]:
        identities = [(lookup[task][seed]["source_episode"], lookup[task][seed]["source_start_row"],
            lookup[task][seed]["methods"]["dispatch"]["execution_identity"]["initial"]) for task in TASKS]
        if not all(value == identities[0] for value in identities):
            raise ValueError("physical root/goal identity differs across task conditions")
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.report) != 3:
        raise ValueError("all three reports and new immutable output required")
    reports, bindings = {}, []
    for path in args.report:
        report = json.loads(path.read_text())
        task = report["sealed_protocol"]["task"]
        if task in reports:
            raise ValueError("duplicate task input")
        reports[task] = report
        bindings.append({"task": task, "path": str(path.resolve()), "sha256": digest(path)})
    aggregates = audit(reports)
    result = {"schema": "reacher_d2p_certified_task_matrix_audit.v1", "status": "completed", "inputs": bindings,
        "aggregate": aggregates, "gates": {"all_fixed_tasks_retained": True, "identity_joined_all_roots": True,
            "certified_decisions_reproduced": True, "all_fd_action_execution_gates": True},
        "claim_boundary": "Descriptive task-premise diagnostic, unequal compute, not formal P1 or full research closure."}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
