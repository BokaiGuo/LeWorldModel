"""Fresh CUDA full-bank refinement-consistent D2P runtime diagnostic.

All model-non-greedy candidates receive the same fine8 solver as deployment.
No qpos or physical outcome may select a candidate. Preserve all roots and
include same-bank greedy, zero, and real shrinking-horizon CEM-MPC controls.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_refined_dispatch as compiler
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as prefix_backend
from experiments.spectral_reachability import preflight_reacher_d2p_p0 as fd_backend
from experiments.spectral_reachability import audit_reacher_d2p_actuator_limit_tasks as actuator_tasks
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202629000)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--compiler", choices=("refined", "geometry", "geometry_certified"), default="refined")
    parser.add_argument("--task", choices=tuple(actuator_tasks.TASKS), default="full_drive")
    args = parser.parse_args()
    global compiler
    base_compiler = compiler
    if args.compiler == "geometry":
        from experiments.spectral_reachability import reacher_d2p_geometry_dispatch
        compiler = reacher_d2p_geometry_dispatch
    elif args.compiler == "geometry_certified":
        from experiments.spectral_reachability import reacher_d2p_certified_geometry_dispatch
        compiler = reacher_d2p_certified_geometry_dispatch
    if args.roots < 4:
        raise ValueError("at least four fixed roots required")
    args.policy, output = args.policy.resolve(), args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output and sidecars required")
    seeds = list(range(args.seed, args.seed + args.roots))
    excluded, bindings = bind_exclusions(args.disjoint_reference, seeds)
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    modules = {"auditor": Path(__file__), "compiler": Path(compiler.__file__), "solver": Path(solver.__file__),
        "rescue": Path(rescue.__file__), "context": Path(execution.__file__), "p0": Path(p0.__file__),
        "runner": Path(runner.__file__), "cem": Path(cem.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    modules["base_refinement_compiler"] = Path(base_compiler.__file__)
    modules["prefix_suffix_backend"] = Path(prefix_backend.__file__)
    modules["fd_backend"] = Path(fd_backend.__file__)
    modules["actuator_task_contract"] = Path(actuator_tasks.__file__)
    modules["stage_admission_dependency"] = Path(p0.stage_admission.__file__)
    if args.compiler == "geometry_certified":
        modules["response_backend"] = Path(compiler.geometry.__file__)
        modules["certified_qp"] = Path(compiler.qp.__file__)
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": digest(args.policy / "weights.pt"), "adapter_sha256": digest(args.adapter_artifact), "source_sha256": source_sha,
        "target_delta": [.65, -.65], "policy": compiler.POLICY, "damping": .01, "proposals": 128,
        "non_greedy_margin": 1e-6, "suffix_iterations": 8, "fine_rescue_exponents": [5, 16],
        "model_selection": "refine ALL immediate-non-greedy candidates plus zero/greedy; exact same suffix solver at observed successor",
        "compiler_variant": args.compiler,
        "task": args.task, "raw_actuator_limits": actuator_tasks.TASKS[args.task],
        "past_action_constraints": "actual source history unchanged; task limits start at planning root",
        "proposal_law": "official_source_raw_packets" if args.task == "full_drive" else actuator_tasks.PROPOSAL_LAW,
        "fd_preflight": {"probe_dims": 8, "pairs": 8, "epsilon": .001, "median_relative_error_max": .1},
        "geometry_qualification": None if args.compiler == "refined" else {
            "anchor": "imagined greedy-prefix raw-zero-suffix goal residual", "damping": .01,
            "solver": "projected_1000" if args.compiler == "geometry" else "certified_active_set_box_qp",
            "rule": "endpoint winner AND strictly higher bounded common-residual task response gain than BOTH controls",
            "physical_labels_used": False},
        "cem": {"samples": 128, "steps": 3, "elite": 16, "horizons": [3, 2], "suffix_seed_offset": 50000,
            "predict_samples": 1920, "best_ever": True},
        "bootstrap_seed": args.seed + 911,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Fresh source-episode-disjoint full-bank solver-consistency diagnostic; NOT formal P1 or equal-budget efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Full-bank refined selector and observed-RGB identical fine8 solver, all fixed fresh roots and real CEM-MPC; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    print("Verifying full source SHA and loading official within-episode packet support.", flush=True)
    pool, pool_receipt = shared.load_packet_pool(dataset, expected_sha256=source_sha)
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    full_lower, full_upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    limits = np.asarray(actuator_tasks.TASKS[args.task], np.float32)
    lower, upper = actuator_tasks.task_bounds(mean, std, limits, device=device)
    predict, samples = model.predict, 0

    def counted_predict(emb, act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, -limits, limits)[0]

    def cem_search(tensors, horizon, seed):
        pixels, goal, history = tensors
        best, _, receipt = cem.search_with_incumbent(
            lambda params: p0._cost_fn(model, pixels, goal, history, params, action_lower=lower, action_upper=upper),
            future_horizon=horizon, action_dim=history.shape[-1], device=device, samples=128, steps=3, elite=16, seed=seed)
        return best, receipt

    rows = []
    try:
        for root in roots:
            seed = root["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != root[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("metadata-planned source root changed")
            bank, support = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed, mean=mean, std=std, lower=full_lower, upper=full_upper)
            pixels, goal, history = prepare(context)
            fd = fd_backend._jacobian_and_fd(model, pixels, goal, history, history_len=3, future_horizon=2,
                probe_dims=8, fd_pairs=8, epsilon=.001, action_lower=lower, action_upper=upper, seed=seed, device=device)
            goal_latent = p0._goal_latent(model, goal)
            start_samples, started = samples, time.perf_counter()

            def progress(done, total):
                if done % 16 == 0 or done == total:
                    print(json.dumps({"seed": seed, "compiled_candidates": done, "required_candidates": total}), flush=True)

            first, receipt = compiler.compile_refined_dispatch(model, pixels, goal, history, proposal_bank=bank,
                action_lower=lower, action_upper=upper, damping=.01, non_greedy_margin=1e-6, progress=progress)
            compilation_seconds = time.perf_counter() - started
            compilation_samples = samples - start_samples
            decision = receipt["dispatch"]
            gates = {"compilation_predict_samples_exact": compilation_samples == receipt["model_predict_sample_evaluations"],
                "no_physical_selection_labels": decision["physical_labels_used"] is False,
                "task_jacobian_fd_preflight": fd["median_relative_error"] <= .1}
            greedy_index = decision["same_bank_greedy_index"]
            before_cem = samples
            cem_full, cem_first_receipt = cem_search((pixels, goal, history), 3, seed)
            initial_cem_samples = samples - before_cem
            prefixes = {"dispatch": first, "same_bank_greedy": bank[greedy_index:greedy_index + 1], "zero": bank[:1], "cem_mpc": cem_full[:, :1]}
            methods = {}
            for name, prefix_params in prefixes.items():
                prefix = raw(prefix_params)
                frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=frame)
                physics = env.unwrapped.env.physics
                after_prefix_distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                next_pixels, next_goal, next_history = prepare(advanced)
                emb = p0._latent_context(model, next_pixels, next_history)
                observed_first_cost = float((emb[0, -1] - goal_latent).square().mean())
                before_suffix, started_suffix = samples, time.perf_counter()
                if name == "cem_mpc":
                    suffix, suffix_receipt = cem_search((next_pixels, next_goal, next_history), 2, seed + 50000)
                    vjp = 0
                    first_samples = initial_cem_samples
                else:
                    suffix, suffix_receipt = solver.compile_snapshots(model, goal_latent, emb, next_history,
                        future_horizon=2, action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,),
                        accept_update=rescue.accept_with_fine_rescue)[8]
                    vjp = suffix_receipt["backward_vjp_rows"]
                    first_samples = compilation_samples if name == "dispatch" else (128 if name == "same_bank_greedy" else 0)
                    gates[f"{name}_suffix_predict_exact"] = samples - before_suffix == suffix_receipt["model_predict_sample_evaluations"]
                suffix_seconds = time.perf_counter() - started_suffix
                suffix_raw = raw(suffix)
                physical, successor_identity = p0._execute(env, seed, suffix_raw, expected_frame=advanced["root_frame"],
                    expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"])
                methods[name] = {"physical": physical, "after_first_chunk_qpos_distance": after_prefix_distance,
                    "observed_first_rgb_goal_cost": observed_first_cost, "first_action_witness_raw": prefix.tolist(),
                    "action_witness_raw": np.concatenate((prefix, suffix_raw)).tolist(),
                    "suffix_receipt": suffix_receipt, "execution_identity": {"initial": identity, "successor": successor_identity},
                    "actual_compute": {"model_predict_sample_evaluations": first_samples + samples - before_suffix,
                        "logical_backward_vjp_rows": vjp + (receipt["backward_vjp_rows"] if name == "dispatch" else 0),
                        "suffix_wall_seconds": suffix_seconds, "dispatch_compilation_wall_seconds": compilation_seconds if name == "dispatch" else None}}
                if name == "cem_mpc":
                    gates["true_cem_mpc_predict_samples_exact"] = methods[name]["actual_compute"]["model_predict_sample_evaluations"] == 1920
                    methods[name]["first_receipt"] = cem_first_receipt
            if not decision["reconfiguration_witness_found"]:
                gates["fallback_exact_greedy_actions"] = methods["dispatch"]["action_witness_raw"] == methods["same_bank_greedy"]["action_witness_raw"]
                gates["fallback_exact_greedy_distance"] = methods["dispatch"]["physical"]["final_qpos_distance"] == methods["same_bank_greedy"]["physical"]["final_qpos_distance"]
            gates["all_root_identities_equal"] = all(method["execution_identity"]["initial"] == methods["dispatch"]["execution_identity"]["initial"] for method in methods.values())
            gates["all_distances_finite"] = all(np.isfinite(method["physical"]["final_qpos_distance"]) for method in methods.values())
            gates["all_future_actions_obey_task_limits"] = all(actuator_tasks.action_limit_gate(method, limits) for method in methods.values())
            row = {**root, "task": args.task, "fd_preflight": fd, "compilation": receipt, "methods": methods, "proposal_support": support, "gates": gates,
                "mechanism": {"actual_rgb_non_greedy_vs_zero": methods["dispatch"]["observed_first_rgb_goal_cost"] >= methods["zero"]["observed_first_rgb_goal_cost"] + 1e-6,
                    "actual_qpos_non_greedy_vs_zero": methods["dispatch"]["after_first_chunk_qpos_distance"] >= methods["zero"]["after_first_chunk_qpos_distance"] + 1e-6}}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "all_gates": all(gates.values()), "decision": decision, "mechanism": row["mechanism"],
                "actual_final": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()}}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("physical integrity gate failed; partial evidence retained")
    finally:
        model.predict = predict
        env.close()
    eligible = [row for row in rows if row["compilation"]["dispatch"]["reconfiguration_witness_found"]]
    aggregate = {"witness_count": len(eligible), "eligible_actual_rgb_non_greedy": sum(row["mechanism"]["actual_rgb_non_greedy_vs_zero"] for row in eligible),
        "eligible_actual_qpos_non_greedy": sum(row["mechanism"]["actual_qpos_non_greedy_vs_zero"] for row in eligible),
        "paired": {name: runner._paired(rows, "dispatch", name, protocol["bootstrap_seed"]) for name in ("same_bank_greedy", "zero", "cem_mpc")}}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots_retained": [row["seed"] for row in rows] == seeds,
        "all_source_episodes_disjoint": not ({row["source_episode"] for row in rows} & excluded),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_full_bank_refined_dispatch.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "source_pool_receipt": pool_receipt,
        "aggregate": aggregate, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
