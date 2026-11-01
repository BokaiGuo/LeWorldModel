"""Frozen-model, source-disjoint task pilot with explicit actuator limits.

Full drive and two fixed 20x torque-restricted tasks share root/goal/history
and source-derived parameter coordinates. Future physical packets, Jacobians,
selection and execution use the SAME restricted action chart. Past history
remains the actual unconstrained source history. This is not official-source
packet support, a proved controllability-gap family, or formal P1 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_source_supported_p1 as source
from experiments.spectral_reachability import preflight_reacher_d2p_p0 as preflight
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_program_dispatch as dispatch
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots, root_execution_gates

TASKS = {"full_drive": (1., 1.), "weak_shoulder": (.05, 1.), "weak_elbow": (1., .05)}
PROPOSAL_LAW = "actuator_limited_source_parameter_coordinates_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_bounds(mean, std, limits, *, device):
    limits = np.asarray(limits, np.float32)
    if limits.shape != (2,) or not np.isfinite(limits).all() or np.any(limits <= 0) or np.any(limits > 1):
        raise ValueError("two strictly positive bounded actuator limits required")
    lower, upper = shared._normalized_action_bounds(mean, std, -limits, limits)
    return torch.as_tensor(lower, device=device), torch.as_tensor(upper, device=device)


def action_limit_gate(method, limits):
    raw = np.asarray(method["action_witness_raw"], np.float64)
    limits = np.asarray(limits, np.float64)
    if raw.shape != (15, 2) or limits.shape != (2,) or not np.isfinite(raw).all():
        return False
    return bool(np.all(np.abs(raw) <= limits[None] + 1e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202627000)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.roots < 4 or any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise ValueError("at least four roots and new immutable outputs required")
    seeds = list(range(args.seed, args.seed + args.roots))
    excluded, bindings = source._disjoint_reference_binding(args.disjoint_reference, set(seeds))
    dataset, source_hash = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    implementations = {"auditor": Path(__file__), "dispatch": Path(dispatch.__file__), "selector": Path(p1.__file__),
        "runner": Path(runner.__file__), "p0": Path(p0.__file__), "preflight": Path(preflight.__file__),
        "context": Path(execution.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    sealed = {"root_seeds": seeds, "source_roots": roots, "tasks": TASKS, "proposal_law": PROPOSAL_LAW,
        "root_context": "official_source_state_history", "past_action_constraints": "actual original source actions; task limits start at planning root",
        "goal_delta": [.65, -.65], "future_horizon": 2, "prefix_horizon": 1, "raw_steps": 15,
        "proposals": 128, "damping": .01, "non_greedy_margin": 1e-6, "reachability_margin": 0.,
        "future_reference": "same_bank_greedy", "suffix_solver": p1.SUFFIX_SOLVER,
        "fd_probe_dims": 8, "fd_pairs": 8, "fd_epsilon": 1e-3, "fd_median_relative_error_max": .10,
        "source_sha256": source_hash, "weights_sha256": digest(args.policy / "weights.pt"),
        "adapter_sha256": digest(args.adapter_artifact), "disjoint_reference_bindings": bindings,
        "excluded_source_episodes": sorted(excluded), "implementation_sha256": {key: digest(path) for key, path in implementations.items()},
        "scope": "Three fixed task conditions, source-episode-disjoint pilot. No task/limit/threshold selection from physical outcomes, no formal P1 or full compute-matched efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in implementations.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Testing full drive and two fixed torque-limit tasks on all new roots; CUDA only, no learned modules or efficacy claims.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    print("Verifying full official-source SHA256 and loading parameter-coordinate support.", flush=True)
    pool, support = shared.load_packet_pool(dataset, expected_sha256=source_hash)
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
    full_lower, full_upper = task_bounds(mean, std, TASKS["full_drive"], device=device)
    rows = []
    try:
        for root in roots:
            seed = root["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != root[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("predeclared source root changed")
            bank, source_coordinates = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed,
                mean=mean, std=std, lower=full_lower, upper=full_upper)
            source_coordinates = {"coordinate_generation": source_coordinates,
                "actual_future_packet_support": PROPOSAL_LAW, "official_source_raw_packet_claim": False}
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], np.float32), device)
            for task, limits in TASKS.items():
                lower, upper = task_bounds(mean, std, limits, device=device)
                jacobian = preflight._jacobian_and_fd(model, pixels, goal, history, history_len=3,
                    future_horizon=2, probe_dims=8, fd_pairs=8, epsilon=1e-3,
                    action_lower=lower, action_upper=upper, seed=seed, device=device)

                def selected(pixels, goal, history, root_seed):
                    return dispatch.compile_program_dispatch(model, pixels, goal, history, proposal_bank=bank,
                        reconfiguration_horizon=1, future_horizon=2, action_lower=lower, action_upper=upper,
                        damping=.01, proposals=128, non_greedy_margin=1e-6, reachability_margin=0., seed=root_seed,
                        future_reference="same_bank_greedy", proposal_distribution=PROPOSAL_LAW)

                def greedy(pixels, goal, history, _seed):
                    return runner._source_matched_greedy_program(model, pixels, goal, history, proposal_bank=bank,
                        action_lower=lower, action_upper=upper)

                def zero(_pixels, _goal, history, _seed):
                    return torch.zeros((1, 1, history.shape[-1]), device=device), {"program_type": "zero_prefix_control"}

                methods = {}
                for name, program in (("dispatch", selected), ("same_bank_greedy", greedy), ("zero", zero)):
                    methods[name] = runner._two_stage(name=name, first_program=program, env=env, seed=seed, context=context,
                        preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                        action_block=5, progress_horizon=2, damping=.01, device=device)
                gates = {**root_execution_gates(methods), "all_actions_obey_task_limits": all(action_limit_gate(method, limits) for method in methods.values()),
                    "jacobian_fd_within_declared_error": jacobian["median_relative_error"] <= .10}
                row = {**root, "task": task, "raw_torque_limits": limits, "source_coordinate_support": source_coordinates,
                    "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                    "jacobian": jacobian, "methods": methods, "gates": gates}
                with (output.with_suffix(".roots") / f"{task}_root_{seed}.json").open("x") as handle:
                    handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
                rows.append(row)
                print(json.dumps({"seed": seed, "task": task, "gates": gates,
                    "route": methods["dispatch"]["first_receipt"]["dispatch"]["route"],
                    "fd_median_relative_error": jacobian["median_relative_error"],
                    "final_qpos_distance": {name: method["physical"]["final_qpos_distance"] for name, method in methods.items()}}), flush=True)
                if not all(value for key, value in gates.items() if key != "jacobian_fd_within_declared_error"):
                    raise RuntimeError("task physical action/replay contract failed; partial receipt retained")
    finally:
        env.close()
    task_results = {}
    for task in TASKS:
        subset = [row for row in rows if row["task"] == task]
        eligible = [row for row in subset if row["methods"]["dispatch"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"]]
        zero_reach = [row["methods"]["dispatch"]["first_receipt"]["reconfiguration_compilation"]["proposal_rows"][0] for row in subset]
        task_results[task] = {"root_count": len(subset), "model_eligible_roots": len(eligible),
            "zero_prefix_mean_unreachable_residual_fraction": float(np.mean([reach["first_order_unreachable_residual_fraction"] for reach in zero_reach])),
            "zero_prefix_mean_jacobian_norm": float(np.mean([reach["jacobian_frobenius_norm"] for reach in zero_reach])),
            "paired_vs_greedy": runner._paired(subset, "dispatch", "same_bank_greedy", args.seed + 771),
            "paired_vs_zero": runner._paired(subset, "dispatch", "zero", args.seed + 771),
            "eligible_observed_rgb_non_greedy_vs_zero": sum(row["methods"]["dispatch"]["observed_first_chunk"]["candidate_goal_cost"] >= row["methods"]["zero"]["observed_first_chunk"]["candidate_goal_cost"] - 1e-8 for row in eligible),
            "eligible_qpos_non_progress_vs_initial_root": sum(row["methods"]["dispatch"]["physical"]["after_first_chunk_qpos_distance"] >= row["methods"]["dispatch"]["physical"]["initial_qpos_distance"] for row in eligible)}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_root_task_cells_retained": {(row["seed"], row["task"]) for row in rows} == {(seed, task) for seed in seeds for task in TASKS} and len(rows) == len(seeds) * len(TASKS),
        "all_execution_and_fd_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == sealed["implementation_sha256"][key] for key, path in implementations.items())}
    result = {"schema": "reacher_d2p_actuator_limit_task_pilot.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": sealed, "gates": gates, "source_support": support,
        "aggregate": task_results, "rows": rows, "claim_boundary": sealed["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": task_results}), flush=True)


if __name__ == "__main__":
    main()
