"""Recompute identical CUDA response matrices, isolate box-QP numerical error.

Preserve all reference endpoint winners and both controls. Reproduce original
fixed-gradient metrics before comparing certified active-set results. No
physical outcomes, threshold fitting or new root sampling in this audit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as compiler
from experiments.spectral_reachability import bounded_response_qp as certified
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_geometry_qualification import fixed_decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    fixed_decisions(report)
    reference, output = report["sealed_protocol"], args.output.resolve()
    modules = {"auditor": Path(__file__), "response_backend": Path(compiler.__file__), "certified_qp": Path(certified.__file__),
        "p0": Path(compiler.p0.__file__), "prefix_suffix_backend": Path(compiler.p1.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for key in ("p0", "prefix_suffix_backend", "source_context"):
        if digest(modules[key]) != reference["implementation_sha256"][key]:
            raise ValueError("reference model state/scoring implementation changed: " + key)
    if digest(args.policy / "weights.pt") != reference["weights_sha256"] or digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("frozen checkpoint or source receipt changed")
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output required")
    protocol = {"reference_sha256": digest(args.reference), "root_seeds": sorted(reference["root_seeds"]),
        "weights_sha256": reference["weights_sha256"], "adapter_sha256": reference["adapter_sha256"], "source_sha256": reference["source_sha256"],
        "old_gradient_steps": 1000, "active_set_cap": 500, "damping": .01,
        "gain_certificate_tolerance": "1e-8 times max(1, squared common residual norm)",
        "comparison": "candidate feasible gain lower bound strictly exceeds BOTH control gain upper bounds",
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root numerical isolation on identical model action-response matrices; no physical outcome or P1 conclusion."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / (key + ".py"))
    print("Isolating fixed-gradient box-QP error against certified active-set solve, identical CUDA matrices, all reference candidates and roots.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != reference["source_sha256"]:
        raise ValueError("source SHA receipt changed")
    print("Verifying full source SHA and exact source proposal bank.", flush=True)
    pool, pool_receipt = shared.load_packet_pool(dataset, expected_sha256=source_sha)
    if pool_receipt != report["source_pool_receipt"]:
        raise ValueError("source packet pool changed")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    rows = []
    try:
        for source in sorted(report["rows"], key=lambda row: row["seed"]):
            seed, receipt = source["seed"], source["compilation"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            identity = source["methods"]["dispatch"]["execution_identity"]["initial"]
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != identity["reset_rgb_sha256"] or context["goal_rgb_sha256"] != identity["goal_rgb_sha256"]:
                raise RuntimeError("reference source/root/goal changed")
            bank, support = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed, mean=mean, std=std, lower=lower, upper=upper)
            if support != source["proposal_support"]:
                raise RuntimeError("source proposal bank changed")
            pixels, goal, history = compiler.p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
            matrices = []

            def compare(response, residual, **settings):
                old = compiler.bounded_response_gain(response, residual, **settings)
                new = certified.certified_response_gain(response, residual, **settings)
                matrices.append(response.tolist())
                return {**new, "old_gradient_metric": old,
                    "certified_minus_gradient_gain": new["bounded_quadratic_gain"] - old["bounded_quadratic_gain"]}

            geometry, residual, samples, vjp = compiler.response_bank_metrics(model, pixels, goal, history, proposal_bank=bank,
                action_lower=lower, action_upper=upper, required=receipt["dispatch"]["geometry_required_indices"],
                greedy=receipt["dispatch"]["same_bank_greedy_index"], damping=.01, gain_solver=compare)
            old_metrics = receipt["common_task_response"]["candidate_metrics"]
            gates = {"old_common_residual_replayed_exact": residual.tolist() == receipt["common_task_response"]["residual"],
                "all_original_fixed_gradient_gains_replayed_exact": all(cell["old_gradient_metric"]["bounded_quadratic_gain"] == old_metrics[str(index)]["bounded_quadratic_gain"] for index, cell in geometry.items()),
                "all_old_zero_endpoints_replayed_exact": all(cell["zero_suffix_endpoint"] == old_metrics[str(index)]["zero_suffix_endpoint"] for index, cell in geometry.items()),
                "all_new_objective_certificates_pass": all(cell["objective_certificate_passed"] for cell in geometry.values()),
                "certified_objective_not_worse_than_gradient": all(cell["certified_minus_gradient_gain"] >= -1e-8 for cell in geometry.values())}
            decision = certified.qualify_certified_decision(receipt, geometry)
            row = {"seed": seed, "source_episode": source["source_episode"], "old_decision": receipt["dispatch"], "certified_decision": decision,
                "common_residual": residual.tolist(), "metrics": geometry, "response_matrices_in_required_index_order": matrices,
                "required_indices": receipt["dispatch"]["geometry_required_indices"], "predict_samples": samples, "backward_vjp_rows": vjp, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates": gates, "old_index": receipt["dispatch"]["selected_program_index"],
                "certified_index": decision["selected_program_index"], "max_gain_correction": max(cell["certified_minus_gradient_gain"] for cell in geometry.values())}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("same-matrix isolation gate failed; partial evidence retained")
    finally:
        env.close()
    aggregate = {"matrices": sum(len(row["metrics"]) for row in rows),
        "changed_decisions": sum(row["old_decision"]["selected_program_index"] != row["certified_decision"]["selected_program_index"] for row in rows),
        "certified_witnesses": sum(row["certified_decision"]["reconfiguration_witness_found"] for row in rows),
        "max_gain_correction": max(cell["certified_minus_gradient_gain"] for row in rows for cell in row["metrics"].values()),
        "max_objective_duality_gap_bound": max(cell["qp_certificate"]["objective_duality_gap_upper_bound"] for row in rows for cell in row["metrics"].values())}
    gates = {"all_roots_retained": [row["seed"] for row in rows] == sorted(reference["root_seeds"]),
        "all_isolation_gates": all(all(row["gates"].values()) for row in rows), "cuda_only": True,
        "weights_unchanged": before == shared._parameter_hash(model), "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_response_qp_convergence.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
