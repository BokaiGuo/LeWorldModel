"""Fixed-root causal replay: endpoint-only versus common-response qualification.

The two decisions were sealed in the original compiler receipt before any
branch outcomes. No new candidate choice, margin tuning or root filtering.
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
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as compiler
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_post_prefix_geometry import validate_reference


def fixed_decisions(report):
    validate_reference(report)
    if report["sealed_protocol"].get("compiler_variant") != "geometry":
        raise ValueError("geometry-qualified reference required")
    result = {}
    for row in report["rows"]:
        receipt = row["compilation"]
        geometry = {int(key): value for key, value in receipt["common_task_response"]["candidate_metrics"].items()}
        decision = compiler.qualify_decision(receipt, geometry)
        if decision != receipt["dispatch"]:
            raise ValueError("stored common-response decision cannot be reproduced")
        endpoint = compiler.base.select_refined_bank(receipt["proposal_rows"], non_greedy_margin=receipt["non_greedy_margin"])
        if endpoint != receipt["endpoint_only_dispatch"]:
            raise ValueError("stored endpoint-only decision cannot be reproduced")
        result[row["seed"]] = {"dispatch": decision["selected_program_index"], "endpoint_only": endpoint["selected_program_index"],
            "same_bank_greedy": decision["same_bank_greedy_index"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    decisions = fixed_decisions(report)
    reference, output = report["sealed_protocol"], args.output.resolve()
    if digest(args.policy / "weights.pt") != reference["weights_sha256"] or digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("frozen model or source receipt changed")
    modules = {"auditor": Path(__file__), "compiler": Path(compiler.__file__), "base_refinement_compiler": Path(compiler.base.__file__),
        "context": Path(execution.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "p0": Path(p0.__file__), "runner": Path(runner.__file__), "prefix_suffix_backend": Path(compiler.p1.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for key, path in modules.items():
        if key != "auditor" and digest(path) != reference["implementation_sha256"][key]:
            raise ValueError("reference implementation changed: " + key)
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"reference_sha256": digest(args.reference), "root_seeds": sorted(decisions), "fixed_decisions": decisions,
        "weights_sha256": reference["weights_sha256"], "adapter_sha256": reference["adapter_sha256"], "source_sha256": reference["source_sha256"],
        "implementation_sha256": {key: digest(path) for key, path in modules.items()}, "bootstrap_seed": 202631911,
        "scope": "All-known-root fixed-decision causal replay, not fresh efficacy, equal-budget comparison or P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / (key + ".py"))
    print("Same-root fixed endpoint-only versus common-response decisions, exact observed fine8 deployment; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != reference["source_sha256"]:
        raise ValueError("source SHA receipt changed")
    print("Verifying full source SHA and exact official packet pool.", flush=True)
    pool, pool_receipt = shared.load_packet_pool(dataset, expected_sha256=source_sha)
    if pool_receipt != report["source_pool_receipt"]:
        raise ValueError("official packet pool changed")
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
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            expected = source["methods"]["dispatch"]["execution_identity"]["initial"]
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != expected["reset_rgb_sha256"] or context["goal_rgb_sha256"] != expected["goal_rgb_sha256"]:
                raise RuntimeError("reference source/root/goal changed")
            bank, support = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed, mean=mean, std=std, lower=lower, upper=upper)
            if support != source["proposal_support"]:
                raise RuntimeError("reference proposal bank changed")
            methods, gates, cache = {}, {}, {}
            for name, index in decisions[seed].items():
                if index not in cache:
                    prefix = p0._to_raw_actions(bank[index:index + 1], mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]
                    frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                    identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                    advanced = execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=frame)
                    pixels, goal, history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
                    emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                    params, receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                        action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                    suffix = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]
                    physical, successor = p0._execute(env, seed, suffix, expected_frame=advanced["root_frame"],
                        expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"])
                    cache[index] = {"physical": physical, "first_action_witness_raw": prefix.tolist(),
                        "action_witness_raw": np.concatenate((prefix, suffix)).tolist(), "suffix_receipt": receipt,
                        "execution_identity": {"initial": identity, "successor": successor}}
                methods[name] = cache[index]
                if name in source["methods"]:
                    gates[name + "_actions_exact"] = methods[name]["action_witness_raw"] == source["methods"][name]["action_witness_raw"]
                    gates[name + "_distance_exact"] = methods[name]["physical"]["final_qpos_distance"] == source["methods"][name]["physical"]["final_qpos_distance"]
            row = {"seed": seed, "source_episode": source["source_episode"], "fixed_decisions": decisions[seed], "methods": methods, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates": gates, "indices": decisions[seed],
                "actual_final": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()}}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("exact reference replay failed; partial evidence retained")
    finally:
        env.close()
    gates = {"all_roots_retained": [row["seed"] for row in rows] == sorted(decisions),
        "all_replay_gates": all(all(row["gates"].values()) for row in rows), "cuda_only": True,
        "weights_unchanged": before == shared._parameter_hash(model), "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    aggregate = {"decisions_changed": sum(row["fixed_decisions"]["dispatch"] != row["fixed_decisions"]["endpoint_only"] for row in rows),
        "paired": {name: runner._paired(rows, "dispatch", name, protocol["bootstrap_seed"]) for name in ("endpoint_only", "same_bank_greedy")}}
    result = {"schema": "reacher_d2p_geometry_qualification_causal.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
