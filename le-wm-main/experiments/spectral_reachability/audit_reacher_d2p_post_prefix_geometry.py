"""Evaluator-only physical suffix geometry, never a planner or P1 override.

Replay all frozen reference branches. At each exact post-prefix state, apply
literal raw-zero ten-step futures and both signs of all twenty raw torque
coordinates at scales .1 and .25. Wrapped angular finite differences separate
response geometry from passive drift and from suffix solver quality.
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
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest

SCALES = (.1, .25)
METHODS = ("dispatch", "same_bank_greedy", "zero")


def angular_difference(left, right):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if left.shape != (2,) or right.shape != (2,) or not np.isfinite([left, right]).all():
        raise ValueError("finite two-joint qpos required")
    return (left - right + np.pi) % (2 * np.pi) - np.pi


def geometry_metrics(response, residual, *, damping=.01, steps=1000):
    """Fixed bounded quadratic diagnostic; CUDA in live runs, no oracle control."""
    if response.shape != (2, 20) or residual.shape != (2,):
        raise ValueError("two-by-twenty response and two-joint residual required")
    if response.device != residual.device or response.dtype != residual.dtype:
        raise ValueError("matching dtype and device required")
    if not torch.isfinite(response).all() or not torch.isfinite(residual).all() or damping <= 0 or steps < 1:
        raise ValueError("finite diagnostic and positive fixed optimizer settings required")
    singular = torch.linalg.svdvals(response)
    lipschitz = singular[0].square() + damping
    action = torch.zeros(20, dtype=response.dtype, device=response.device)
    for _ in range(steps):
        gradient = response.T @ (response @ action - residual) + damping * action
        action = (action - gradient / lipschitz).clamp(-1, 1)
    error = residual - response @ action
    baseline = residual.square().sum()
    objective = error.square().sum() + damping * action.square().sum()
    gradient = response.T @ (response @ action - residual) + damping * action
    return {"singular_values": singular.tolist(), "response_frobenius_norm": float(response.norm()),
        "baseline_squared_residual": float(baseline), "bounded_quadratic_objective": float(objective),
        "bounded_quadratic_gain": float(baseline - objective), "linear_residual_norm": float(error.norm()),
        "projected_gradient_residual": float((action - (action - gradient / lipschitz).clamp(-1, 1)).norm()),
        "action_scope": "Evaluator-only bounded linear surrogate, not deployed or fed back into planner."}


def validate_reference(report):
    if report.get("schema") != "reacher_d2p_full_bank_refined_dispatch.v1" or report.get("status") != "completed":
        raise ValueError("completed full-bank refined dispatch receipt required")
    if not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("reference integrity gates must pass")
    seeds = [row["seed"] for row in report["rows"]]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(report["sealed_protocol"]["root_seeds"]):
        raise ValueError("duplicate-free root identity set equality required")
    for row in report["rows"]:
        for name in METHODS:
            method = row["methods"][name]
            actions = np.asarray(method["action_witness_raw"], np.float32)
            # Match _to_raw_actions' existing float32 inverse-normalization
            # roundoff contract. Replay stored bytes, never clip the witness.
            if actions.shape != (15, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1 + 1e-6:
                raise ValueError("full finite bounded fifteen-step witness required")
            if not np.array_equal(actions[:5], np.asarray(method["first_action_witness_raw"], np.float32)):
                raise ValueError("stored prefix and full action witness disagree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    validate_reference(report)
    reference, output = report["sealed_protocol"], args.output.resolve()
    implementations = {"auditor": Path(__file__), "context": Path(execution.__file__), "p0": Path(p0.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for key in ("context", "source_context", "p0"):
        if digest(implementations[key]) != reference["implementation_sha256"][key]:
            raise ValueError("reference execution implementation changed")
    if digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("reference source receipt changed")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != reference["source_sha256"]:
        raise ValueError("source receipt SHA changed")
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"root_seeds": sorted(reference["root_seeds"]), "reference_sha256": digest(args.reference),
        "source_sha256_receipt": source_sha, "methods": METHODS, "raw_probe_scales": SCALES,
        "task": reference.get("task", "full_drive"), "raw_actuator_limits": reference.get("raw_actuator_limits", [1., 1.]),
        "response_coordinates": "unit bounded actuator commands; each raw torque probe is scale times its actuator limit",
        "suffix_raw_steps": 10, "coordinates": 20, "signs": [-1, 1], "damping": .01, "projected_steps": 1000,
        "stored_action_float32_bound_tolerance": 1e-6, "stored_actions_clipped": False,
        "root_identity": "sealed_observation_exact_state; exact integration and scene, observed RGB not synthesized",
        "residuals": "own raw-zero-future residual and common greedy raw-zero-future residual; wrapped target minus qpos",
        "implementation_sha256": {key: digest(path) for key, path in implementations.items()},
        "scope": "All-known-root physical mechanism diagnosis; evaluator qpos never enters planning. Finite-scale estimates, not a controllability theorem, selector or P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in implementations.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / (key + ".py"))
    print("All-root post-prefix physical response, both scales and twenty coordinates; evaluator only, CUDA geometry, no neural CPU fallback.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["target_delta"]))
    actuator_limits = np.asarray(protocol["raw_actuator_limits"], np.float32)
    if actuator_limits.shape != (2,) or not np.isfinite(actuator_limits).all() or np.any(actuator_limits <= 0) or np.any(actuator_limits > 1):
        raise ValueError("finite positive physical actuator limits required")
    rows = []
    try:
        for source in sorted(report["rows"], key=lambda row: row["seed"]):
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            expected = source["methods"]["dispatch"]["execution_identity"]["initial"]
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != expected["reset_rgb_sha256"] or context["goal_rgb_sha256"] != expected["goal_rgb_sha256"]:
                raise RuntimeError("reference source/root/goal changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            initial_root_distance = p0._qpos_distance(np.asarray(context["execution_snapshot"]["qpos"]),
                np.asarray(context["execution_snapshot"]["target_qpos"]))
            cells, gates, cache = {}, {}, {}
            for name in METHODS:
                actions = np.asarray(source["methods"][name]["action_witness_raw"], np.float32)
                prefix = actions[:5]
                cache_key = actions.tobytes()
                if cache_key in cache:
                    cell = dict(cache[cache_key])
                    cell["duplicate_prefix_cached"] = True
                else:
                    root, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    p0._verify_execution_identity(root, goal, context["root_frame"], context["goal"])
                    advanced = execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=root)
                    snapshot = advanced["execution_snapshot"]
                    snapshot["observed_root_rgb"] = execution.pack_rgb_observation(advanced["root_frame"])
                    target = np.asarray(snapshot["target_qpos"], np.float64)[:2]

                    def simulate(future):
                        execution.restore_execution_snapshot(env, seed=seed, snapshot=snapshot, root_identity="sealed_observation_exact_state")
                        execution._execute_raw_steps(env, future)
                        return np.asarray(env.unwrapped.env.physics.data.qpos, np.float64)[:2].copy()

                    zero_qpos = simulate(np.zeros((10, 2), np.float32))
                    probes = []
                    for scale in SCALES:
                        columns, endpoints = [], []
                        for index in range(20):
                            plus = np.zeros((10, 2), np.float32)
                            plus.flat[index] = scale * actuator_limits[index % 2]
                            positive, negative = simulate(plus), simulate(-plus)
                            command_step = float(plus.flat[index]) / float(actuator_limits[index % 2])
                            columns.append(angular_difference(positive, negative) / (2 * command_step))
                            endpoints.append({"coordinate": index, "plus_qpos": positive.tolist(), "minus_qpos": negative.tolist()})
                        response = np.stack(columns, axis=1)
                        probes.append({"scale": scale, "physical_response": response.tolist(), "endpoints": endpoints})
                    cell = {"post_prefix_qpos": snapshot["qpos"], "post_prefix_qvel": snapshot["qvel"],
                        "target_qpos": target.tolist(), "raw_zero_future_qpos": zero_qpos.tolist(),
                        "raw_zero_future_distance": float(np.linalg.norm(angular_difference(target, zero_qpos))),
                        "optimized_final_qpos": simulate(actions[5:]).tolist(), "probes": probes, "duplicate_prefix_cached": False}
                    cache[cache_key] = cell
                final_distance = p0._qpos_distance(np.asarray(cell["optimized_final_qpos"]), np.asarray(cell["target_qpos"]))
                gates[name + "_optimized_witness_replay_exact"] = final_distance == source["methods"][name]["physical"]["final_qpos_distance"]
                gates[name + "_prefix_distance_replay_exact"] = p0._qpos_distance(np.asarray(cell["post_prefix_qpos"]), np.asarray(cell["target_qpos"])) == source["methods"][name]["after_first_chunk_qpos_distance"]
                cell["optimized_final_distance"] = final_distance
                cells[name] = cell
                print(json.dumps({"seed": seed, "probed_method": name, "cached": cell["duplicate_prefix_cached"]}), flush=True)
            common = angular_difference(cells["same_bank_greedy"]["target_qpos"], cells["same_bank_greedy"]["raw_zero_future_qpos"])
            for name, cell in cells.items():
                own = angular_difference(cell["target_qpos"], cell["raw_zero_future_qpos"])
                for probe in cell["probes"]:
                    response = torch.as_tensor(probe["physical_response"], device="cuda", dtype=torch.float64)
                    probe["own_residual_metrics"] = geometry_metrics(response, torch.as_tensor(own, device="cuda", dtype=torch.float64))
                    probe["common_greedy_residual_metrics"] = geometry_metrics(response, torch.as_tensor(common, device="cuda", dtype=torch.float64))
                    gates[name + "_finite_scale_" + str(probe["scale"])] = bool(torch.isfinite(response).all())
            row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                "initial_root_qpos_distance": initial_root_distance,
                "actual_qpos_non_progress_vs_initial_root": p0._qpos_distance(np.asarray(cells["dispatch"]["post_prefix_qpos"]),
                    np.asarray(cells["dispatch"]["target_qpos"])) >= initial_root_distance,
                "model_eligible": source["compilation"]["dispatch"]["reconfiguration_witness_found"],
                "reference_mechanism": source["mechanism"], "methods": cells, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates": gates}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("reference replay or response gate failed; partial evidence retained")
    finally:
        env.close()
    aggregate = {}
    for index, scale in enumerate(SCALES):
        comparisons = []
        for row in rows:
            dispatch, greedy = (row["methods"][name] for name in ("dispatch", "same_bank_greedy"))
            d = dispatch["probes"][index]["common_greedy_residual_metrics"]
            g = greedy["probes"][index]["common_greedy_residual_metrics"]
            comparisons.append({"seed": row["seed"], "model_eligible": row["model_eligible"],
                "common_residual_bounded_gain_change": d["bounded_quadratic_gain"] - g["bounded_quadratic_gain"],
                "smallest_singular_value_change": d["singular_values"][-1] - g["singular_values"][-1],
                "passive_distance_change": dispatch["raw_zero_future_distance"] - greedy["raw_zero_future_distance"],
                "optimized_distance_change": dispatch["optimized_final_distance"] - greedy["optimized_final_distance"]})
        eligible = [cell for cell in comparisons if cell["model_eligible"]]
        aggregate[str(scale)] = {"comparisons": comparisons, "eligible_count": len(eligible),
            "eligible_common_residual_gain_positive": sum(cell["common_residual_bounded_gain_change"] > 0 for cell in eligible),
            "eligible_smallest_singular_value_positive": sum(cell["smallest_singular_value_change"] > 0 for cell in eligible)}
    gates = {"all_reference_roots_retained": [row["seed"] for row in rows] == sorted(reference["root_seeds"]),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows), "cuda_geometry_only": True,
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in implementations.items())}
    result = {"schema": "reacher_d2p_post_prefix_physical_geometry.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "gates": gates, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior full source SHA receipt reused; actual source root/goal identities and complete action witnesses rechecked. No new full source SHA claimed.",
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
