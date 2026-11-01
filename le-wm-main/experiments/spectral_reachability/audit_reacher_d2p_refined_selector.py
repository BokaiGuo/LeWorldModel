"""Same-root causal audit of refinement-consistent reconfiguration scoring.

Preserve every old candidate/control; decide from imagined RGB dynamics before
executing any branch. Then replay observed-RGB fine8 suffixes. This restricted
three-branch audit is not a full-bank selector, fresh efficacy, or P1 admission.
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
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_refined_reconfiguration as selector
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest


def validated_rows(report):
    if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("complete integrity-passing reference required")
    seeds = report["sealed_protocol"]["root_seeds"]
    rows = report["rows"]
    lookup = {row["seed"]: row for row in rows}
    if not seeds or len(seeds) != len(set(seeds)) or len(lookup) != len(rows) or set(lookup) != set(seeds):
        raise ValueError("complete unique root identity sets required")
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-reference", type=Path, action="append", required=True)
    parser.add_argument("--refined-reference", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.runtime_reference) != len(args.refined_reference):
        raise ValueError("paired runtime/refined reference files required")
    inputs, seen = [], set()
    for runtime_path, refined_path in zip(args.runtime_reference, args.refined_reference):
        runtime, refined = json.loads(runtime_path.read_text()), json.loads(refined_path.read_text())
        sources, controls = validated_rows(runtime), validated_rows(refined)
        if set(sources) != set(controls) or seen & set(sources):
            raise ValueError("reference panels differ or overlap")
        if runtime["schema"] not in ("reacher_d2p_program_dispatch_diagnostic.v1", "reacher_d2p_suffix_repair_confirmation.v1"):
            raise ValueError("unsupported source runtime schema")
        if refined["schema"] not in ("reacher_d2p_iterated_suffix_audit.v1", "reacher_d2p_suffix_repair_confirmation.v1"):
            raise ValueError("unsupported refined suffix reference schema")
        if refined["schema"] == "reacher_d2p_iterated_suffix_audit.v1" and not refined["sealed_protocol"].get("fine_rescue"):
            raise ValueError("fine-rescue reference required")
        seen.update(sources)
        inputs.append((runtime_path, refined_path, runtime, refined, sources, controls))
    first = inputs[0][2]["sealed_protocol"]
    for _, _, runtime, refined, _, _ in inputs:
        if any(report["sealed_protocol"][key] != first[key] for report in (runtime, refined)
               for key in ("weights_sha256", "adapter_sha256", "source_sha256", "target_delta")):
            raise ValueError("common frozen model, normalizer, source and goal protocol required")
    args.policy = args.policy.resolve()
    if digest(args.policy / "weights.pt") != first["weights_sha256"] or digest(args.adapter_artifact) != first["adapter_sha256"]:
        raise ValueError("reference model or normalizer changed")
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output required")
    modules = {"auditor": Path(__file__), "selector": Path(selector.__file__), "suffix_solver": Path(solver.__file__),
        "rescue": Path(rescue.__file__), "old_selector": Path(p1.__file__), "context": Path(execution.__file__),
        "p0": Path(p0.__file__), "runner": Path(runner.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    protocol = {"root_seeds": sorted(seen), "weights_sha256": first["weights_sha256"], "adapter_sha256": first["adapter_sha256"],
        "source_sha256": first["source_sha256"], "target_delta": first["target_delta"], "damping": .01,
        "suffix_iterations": 8, "non_greedy_margin": 1e-6, "fine_rescue_exponents": [5, 16],
        "references": [{"runtime_path": str(r.resolve()), "runtime_sha256": digest(r),
            "refined_path": str(f.resolve()), "refined_sha256": digest(f)} for r, f, *_ in inputs],
        "model_score_atol": 1e-6, "bootstrap_seed": 202629011,
        "decision_order": "all imagined-state scores and selection frozen before any branch's physical successor",
        "candidate_set": "fixed historical dispatch, same-bank greedy and zero only, not full bank",
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root three-branch selector/solver consistency diagnostic, not fresh efficacy or formal P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Auditing fixed-candidate selector with identical fine8 scoring/execution, before physical outcomes; CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != first["source_sha256"]:
        raise ValueError("source receipt changed")
    print("Verifying full official-source SHA and reconstructing exact source proposal packets.", flush=True)
    pool, pool_receipt = shared.load_packet_pool(dataset, expected_sha256=source_sha)
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(first["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    predict, sample_count = model.predict, 0

    def counted_predict(emb, act):
        nonlocal sample_count
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        sample_count += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    rows = []
    try:
        for _, _, runtime, refined, sources, controls in inputs:
            fresh_layout = runtime["schema"] == "reacher_d2p_suffix_repair_confirmation.v1"
            for seed, source in sorted(sources.items()):
                original = {name: source["methods"][name + "/single" if fresh_layout else name] for name in ("dispatch", "same_bank_greedy")}
                decision = original["dispatch"]["first_receipt"]["dispatch"]
                compilation = original["dispatch"]["first_receipt"]["reconfiguration_compilation"]
                indices = {"dispatch": decision["selected_program_index"], "same_bank_greedy": decision["same_bank_greedy_index"], "zero": 0}
                context = shared._source_state_context(env, dataset=dataset, seed=seed)
                if any(context[key] != source[key] for key in ("source_episode", "source_start_row")):
                    raise RuntimeError("source root identity changed")
                pixels, goal, history = prepare(context)
                goal_latent = p0._goal_latent(model, goal)
                bank, support = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed, mean=mean, std=std, lower=lower, upper=upper)
                if support != source["proposal_support"]:
                    raise RuntimeError("source proposal identity changed")
                model_cells, imagined_cache, gates = {}, {}, {}
                selection_samples = sample_count
                for name, index in indices.items():
                    if index not in imagined_cache:
                        state = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
                        legacy = solver.compile_snapshots(model, goal_latent, *state, future_horizon=2, action_lower=lower,
                            action_upper=upper, damping=.01, checkpoints=(1,))[1]
                        refined_params, receipt = solver.compile_snapshots(model, goal_latent, *state, future_horizon=2,
                            action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                        imagined_cache[index] = {"legacy_cost": legacy[1]["trace"][-1]["cost_after"],
                            "refined_goal_cost": receipt["trace"][-1]["cost_after"], "receipt": receipt,
                            "legacy_receipt": legacy[1], "imagined_suffix_raw": raw(refined_params).tolist()}
                    record = imagined_cache[index]
                    old_row = compilation["proposal_rows"][index]
                    model_cells[name] = {"proposal_index": index, "immediate_goal_cost": old_row["immediate_goal_cost"], **record}
                    gates[f"{name}_legacy_imagined_score_replayed"] = abs(record["legacy_cost"] - old_row["bounded_dls_suffix_goal_cost"]) <= 1e-6
                selection_samples = sample_count - selection_samples
                # The decision is committed before obtaining any new branch RGB.
                rechecked = selector.refine_witness_decision(model_cells, had_witness=decision["reconfiguration_witness_found"])
                methods, physical_cache = {}, {}
                for name, index in indices.items():
                    if index not in physical_cache:
                        prefix = raw(bank[index:index + 1])
                        frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                        identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                        advanced = execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=frame)
                        next_pixels, next_goal, next_history = prepare(advanced)
                        emb = p0._latent_context(model, next_pixels, next_history)
                        params, receipt = solver.compile_snapshots(model, goal_latent, emb, next_history, future_horizon=2,
                            action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                        suffix = raw(params)
                        physical, successor_identity = p0._execute(env, seed, suffix, expected_frame=advanced["root_frame"],
                            expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"])
                        physical_cache[index] = {"physical": physical, "action_witness_raw": np.concatenate((prefix, suffix)).tolist(),
                            "first_action_witness_raw": prefix.tolist(), "actual_first_rgb_goal_cost": float((emb[0, -1] - goal_latent).square().mean()),
                            "observed_suffix_model_cost": receipt["trace"][-1]["cost_after"], "suffix_receipt": receipt,
                            "successor_rgb_sha256": advanced["root_rgb_sha256"], "execution_identity": {"initial": identity, "successor": successor_identity}}
                    methods[name] = physical_cache[index]
                    method = methods[name]
                    if name != "zero":
                        expected = original[name]
                        fine_key = f"{name}/fine_rescue_8" if fresh_layout else f"{name}/dls_8"
                        fine = controls[seed]["methods"][fine_key]
                        gates[f"{name}_prefix_exact"] = method["first_action_witness_raw"] == expected["first_action_witness_raw"]
                        gates[f"{name}_successor_rgb_exact"] = method["successor_rgb_sha256"] == expected["observed_successor_context"]["root_rgb_sha256"]
                        gates[f"{name}_full_fine_actions_exact"] = method["action_witness_raw"] == fine["action_witness_raw"]
                        gates[f"{name}_fine_distance_exact"] = method["physical"]["final_qpos_distance"] == fine["physical"]["final_qpos_distance"]
                        expected_cost = expected["observed_first_chunk"]["candidate_goal_cost"]
                    else:
                        gates["zero_prefix_exact"] = method["first_action_witness_raw"] == original["dispatch"]["zero_prefix_action_witness_raw"]
                        expected_cost = original["dispatch"]["observed_first_chunk"]["zero_prefix_goal_cost"]
                    gates[f"{name}_first_observed_cost_exact"] = method["actual_first_rgb_goal_cost"] == expected_cost
                methods["refined_policy"] = methods[rechecked["selected_branch"]]
                row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                    "old_witness": decision["reconfiguration_witness_found"], "model_cells": model_cells, "decision": rechecked,
                    "selection_compute": {"model_predict_samples": selection_samples,
                        "logical_backward_vjp_rows": sum(value["receipt"]["backward_vjp_rows"] + value["legacy_receipt"]["backward_vjp_rows"] for value in imagined_cache.values())},
                    "methods": methods, "gates": gates}
                with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                    handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
                rows.append(row)
                print(json.dumps({"seed": seed, "all_gates": all(gates.values()), "decision": rechecked,
                    "predicted_refined": {key: value["refined_goal_cost"] for key, value in model_cells.items()},
                    "actual_final": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()}}), flush=True)
                if not all(gates.values()):
                    raise RuntimeError("fixed-branch replay gate failed; partial evidence retained")
    finally:
        model.predict = predict
        env.close()
    rows.sort(key=lambda row: row["seed"])
    witnesses = [row for row in rows if row["old_witness"]]
    aggregate = {"old_witness_count": len(witnesses), "retained_witness_count": sum(row["decision"]["witness_retained"] for row in witnesses),
        "retained_witness_roots": [row["seed"] for row in witnesses if row["decision"]["witness_retained"]],
        "paired": {name: runner._paired(rows, "refined_policy", name, protocol["bootstrap_seed"]) for name in ("dispatch", "same_bank_greedy", "zero")},
        "old_witness_details": [{"seed": row["seed"], "retained": row["decision"]["witness_retained"],
            "predicted_refined_gain": row["decision"]["refined_greedy_gain"],
            "actual_dispatch_minus_greedy": row["methods"]["dispatch"]["physical"]["final_qpos_distance"] - row["methods"]["same_bank_greedy"]["physical"]["final_qpos_distance"]} for row in witnesses]}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_retained": [row["seed"] for row in rows] == sorted(seen),
        "all_replay_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_refined_selector_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "source_pool_receipt": pool_receipt,
        "aggregate": aggregate, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
