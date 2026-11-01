"""Fresh episode-disjoint confirmation of suffix repairs and actual CEM-MPC.

Freeze all variants before new outcomes. Original D2P prefix selection is
unchanged. A real shrinking-horizon CEM executes its own observed-RGB suffix,
not a substituted DLS controller. No formal P1/equal-budget claim is made.
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
from experiments.spectral_reachability import reacher_d2p_program_dispatch as dispatch
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest, fixed_prefix_gate
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability.reacher_d2p_packet_support import SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION


def bind_exclusions(paths, seeds):
    excluded, bindings = set(), []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()) or not report.get("rows"):
            raise ValueError("complete integrity-passing exclusion references required")
        if set(seeds) & {row["seed"] for row in report["rows"]}:
            raise ValueError("root seed overlap")
        excluded.update(row["source_episode"] for row in report["rows"])
        excluded.update(report["sealed_protocol"].get("excluded_source_episodes", []))
        bindings.append({"path": str(path.resolve()), "sha256": digest(path)})
    return excluded, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202628000)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    if args.roots < 4:
        raise ValueError("at least four fixed roots required")
    args.policy, output = args.policy.resolve(), args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output and sidecars required")
    seeds = list(range(args.seed, args.seed + args.roots))
    excluded, bindings = bind_exclusions(args.disjoint_reference, seeds)
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    modules = {"auditor": Path(__file__), "context": Path(execution.__file__), "dispatch": Path(dispatch.__file__),
        "selector": Path(p1.__file__), "p0": Path(p0.__file__), "runner": Path(runner.__file__), "cem": Path(cem.__file__),
        "solver": Path(solver.__file__), "rescue": Path(rescue.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded),
        "exclusion_bindings": bindings, "weights_sha256": digest(args.policy / "weights.pt"),
        "adapter_sha256": digest(args.adapter_artifact), "source_sha256": source_sha,
        "target_delta": [.65, -.65], "damping": .01, "proposals": 128, "future_reference": "same_bank_greedy",
        "suffix_variants": ["original_single_step", "coarse_8", "fine_rescue_8"], "fine_rescue_exponents": [5, 16],
        "cem_mpc": {"initial_horizon": 3, "suffix_horizon": 2, "samples": 128, "steps_per_stage": 3, "elite": 16,
            "output_selection": "best_ever", "suffix_seed_offset": 50000, "predict_samples_per_root": 1920,
            "executes_own_observed_rgb_suffix": True},
        "resource_contract": "All method predict samples and logical VJP rows separate; NOT full compute matched",
        "bootstrap_seed": args.seed + 911, "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Fresh source-episode-disjoint repair confirmation pilot, not formal P1, independent replication, or resource-matched efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Confirming all predeclared suffix variants on new episode-disjoint roots, with real 3+2 CEM-MPC; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    print("Verifying the complete 93GB official source SHA before loading packet support.", flush=True)
    pool, pool_receipt = shared.load_packet_pool(dataset, expected_sha256=source_sha)
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
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

    def advance(context, prefix, seed):
        frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
        p0._verify_execution_identity(frame, goal, context["root_frame"], context["goal"])
        return execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=frame)

    def execute_suffix(advanced, suffix, seed):
        physical, identity = p0._execute(env, seed, suffix, expected_frame=advanced["root_frame"],
            expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"])
        return {"physical": physical, "successor_execution_identity": identity}

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
                raise RuntimeError("metadata-planned root changed")
            bank, bank_receipt = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed,
                mean=mean, std=std, lower=lower, upper=upper)

            def dispatched(pixels, goal, history, root_seed):
                return dispatch.compile_program_dispatch(model, pixels, goal, history, proposal_bank=bank,
                    reconfiguration_horizon=1, future_horizon=2, action_lower=lower, action_upper=upper,
                    damping=.01, proposals=128, non_greedy_margin=1e-6, reachability_margin=0., seed=root_seed,
                    future_reference="same_bank_greedy", proposal_distribution=SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION)

            def greedy(pixels, goal, history, _seed):
                return runner._source_matched_greedy_program(model, pixels, goal, history, proposal_bank=bank,
                    action_lower=lower, action_upper=upper)

            methods, gates = {}, {}
            for name, program in (("dispatch", dispatched), ("same_bank_greedy", greedy)):
                start_samples = sample_count
                original = runner._two_stage(name=name, first_program=program, env=env, seed=seed, context=context,
                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                    action_block=5, progress_horizon=2, damping=.01, device=device)
                original["actual_compute"] = {"model_predict_sample_evaluations": sample_count - start_samples,
                    "logical_backward_vjp_rows": original["first_receipt"].get("backward_vjp_rows", 0) + original["suffix_receipt"]["backward_vjp_rows"]}
                methods[f"{name}/single"] = original
                prefix = np.asarray(original["first_action_witness_raw"], np.float32)
                advanced = advance(context, prefix, seed)
                gates[f"{name}_same_successor_rgb"] = advanced["root_rgb_sha256"] == original["observed_successor_context"]["root_rgb_sha256"]
                pixels, goal, history = prepare(advanced)
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                for variant, callback in (("coarse_8", None), ("fine_rescue_8", rescue.accept_with_fine_rescue)):
                    start_samples = sample_count
                    snapshots = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                        action_lower=lower, action_upper=upper, damping=.01, checkpoints=(1, 8), accept_update=callback)
                    params, receipt = snapshots[8]
                    gates[f"{name}_{variant}_compute_exact"] = sample_count - start_samples == receipt["model_predict_sample_evaluations"]
                    if variant == "coarse_8":
                        gates[f"{name}_single_actions_exact"] = np.concatenate((prefix, raw(snapshots[1][0]))).tolist() == original["action_witness_raw"]
                    suffix = raw(params)
                    actions = np.concatenate((prefix, suffix))
                    method = execute_suffix(advanced, suffix, seed)
                    method.update({"action_witness_raw": actions.tolist(), "suffix_receipt": receipt,
                        "actual_compute": {"model_predict_sample_evaluations": original["actual_compute"]["model_predict_sample_evaluations"] - 14 + receipt["model_predict_sample_evaluations"],
                            "logical_backward_vjp_rows": original["first_receipt"].get("backward_vjp_rows", 0) + receipt["backward_vjp_rows"]},
                        "compute_note": "Deploy cost reuses same first-program cost; diagnostic recomputation excluded and separately stored in solver receipt"})
                    methods[f"{name}/{variant}"] = method
                    gates[f"{name}_{variant}_fixed_prefix"] = fixed_prefix_gate(actions, prefix)
                    gates[f"{name}_{variant}_model_nonascending"] = receipt["cost_nonincreasing"]
            start_samples = sample_count
            full, first_receipt = cem_search(prepare(context), 3, seed)
            prefix = raw(full[:, :1])
            advanced = advance(context, prefix, seed)
            suffix, suffix_receipt = cem_search(prepare(advanced), 2, seed + 50000)
            actual_cem_samples = sample_count - start_samples
            method = execute_suffix(advanced, raw(suffix), seed)
            method.update({"action_witness_raw": np.concatenate((prefix, raw(suffix))).tolist(),
                "initial_full_plan_raw": raw(full).tolist(), "first_receipt": first_receipt, "suffix_receipt": suffix_receipt,
                "actual_compute": {"model_predict_sample_evaluations": actual_cem_samples, "logical_backward_vjp_rows": 0}})
            methods["cem_mpc"] = method
            gates["cem_own_observed_suffix_compute_exact"] = actual_cem_samples == 1920
            gates["cem_fixed_prefix"] = fixed_prefix_gate(method["action_witness_raw"], prefix)
            decision = methods["dispatch/single"]["first_receipt"]["dispatch"]
            gates["no_physical_dispatch_labels"] = decision["physical_labels_used"] is False
            if not decision["reconfiguration_witness_found"]:
                gates["fallback_all_suffix_variants_exact_greedy"] = all(methods[f"dispatch/{variant}"]["action_witness_raw"] == methods[f"same_bank_greedy/{variant}"]["action_witness_raw"] for variant in ("single", "coarse_8", "fine_rescue_8"))
            gates["all_distances_finite"] = all(np.isfinite(method["physical"]["final_qpos_distance"]) for method in methods.values())
            row = {**root, "methods": methods, "proposal_support": bank_receipt, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "all_gates": all(gates.values()), "route": decision["route"],
                "final_distances": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()}}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("root intervention gate failed; retain partial root")
    finally:
        model.predict = predict
        env.close()
    aggregate = {"reconfiguration_count": sum(row["methods"]["dispatch/single"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"] for row in rows),
        "paired": {key: runner._paired(rows, key, key.split("/")[0] + "/single", protocol["bootstrap_seed"])
            for key in ("dispatch/coarse_8", "dispatch/fine_rescue_8", "same_bank_greedy/coarse_8", "same_bank_greedy/fine_rescue_8")},
        "vs_true_cem_mpc": {key: runner._paired(rows, key, "cem_mpc", protocol["bootstrap_seed"])
            for key in ("dispatch/single", "dispatch/coarse_8", "dispatch/fine_rescue_8")}}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots_retained": [row["seed"] for row in rows] == seeds,
        "all_source_episodes_disjoint": not ({row["source_episode"] for row in rows} & excluded),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_suffix_repair_confirmation.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "source_pool_receipt": pool_receipt,
        "aggregate": aggregate, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
