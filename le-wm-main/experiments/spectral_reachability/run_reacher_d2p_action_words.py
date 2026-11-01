"""Fresh-root model-native ordered-word pilot, host CUDA only.

All17 finite words are compiled and physically evaluated at every root. Each
20-step prefix advances FOUR causal RGB/action tokens, followed by the same
observed-RGB fine8 ten-step suffix. Flat six-packet SVD/GN and real six/two
packet CEM have the SAME physical horizon and observation checkpoint.
Planning compute is measured, not asserted equal. No P1 stage is opened.
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
from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import preflight_reacher_d2p_p0 as fd
from experiments.spectral_reachability import reacher_d2p_packet_support as source_support
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202642000)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    if args.roots < 4:
        raise ValueError("at least four predeclared source-disjoint roots required")
    output = args.output.resolve()
    paths = [output, output.with_suffix(".protocol.json"), output.with_suffix(".selection.json"),
             output.with_suffix(".roots"), output.with_suffix(".implementation")]
    if any(path.exists() for path in paths):
        raise FileExistsError("new immutable output and sidecars required")
    seeds = list(range(args.seed, args.seed + args.roots))
    excluded, bindings = bind_exclusions(args.disjoint_reference, seeds)
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    p0, solver, rescue, prefix = words.p0, words.solver, words.rescue, words.p1
    modules = {"auditor": Path(__file__), "words": Path(words.__file__), "p0": Path(p0.__file__),
        "prefix": Path(prefix.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "cem": Path(cem.__file__), "fd": Path(fd.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename),
        "source_support": Path(source_support.__file__), "base_selection": Path(words.refinement.__file__),
        "stage_admission_dependency": Path(p0.stage_admission.__file__)}
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": digest(args.policy / "weights.pt"), "adapter_sha256": digest(args.adapter_artifact), "source_sha256": source_sha,
        "policy": words.POLICY, "task": "full_drive", "target_delta": [.65, -.65], "amplitudes": words.AMPLITUDES,
        "proposals": 17, "prefix_packets": 4, "suffix_packets": 2, "action_block": 5, "physical_raw_steps": 30,
        "selection": "same-bank original non-greedy endpoint-winner selector, fine8 suffix on EVERY word; else immediate greedy",
        "model_selection_uses_physical_labels": False, "all_choices_sealed_before_candidate_execution": True,
        "cem": {"horizons": [6, 2], "samples": 128, "steps": 3, "elite": 16, "predict_samples": 3072, "suffix_seed_offset": 50000},
        "flat_control": "six-packet fine8 SVD/GN, execute first four, observe RGB, fine8 remaining two",
        "damping": .01, "iterations": 8, "partition_atol": 2e-6, "fd_median_relative_error_max": .1,
        "bootstrap_seed": args.seed + 911, "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Fresh source-episode-disjoint finite-action-order pilot, not a Lie bracket theorem, P1 pass or compute-matched efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with paths[1].open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    paths[3].mkdir()
    paths[4].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths[4] / (key + ".py"))
    print("Fresh fixed roots, all17 model-generated ordered words, 30 raw steps per branch. Four observed history updates, flat SVD/GN and full CEM controls; host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    print("Verifying full official source SHA; NO official packet proposal pool is used by this compiler.", flush=True)
    if source_support._sha256(dataset) != source_sha:
        raise ValueError("official source hash changed")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    predict, samples = model.predict, 0

    def counted_predict(emb, act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def cem_search(tensors, horizon, seed):
        pixels, goal, history = tensors
        return cem.search_with_incumbent(lambda params: p0._cost_fn(model, pixels, goal, history, params, action_lower=lower, action_upper=upper),
            future_horizon=horizon, action_dim=history.shape[-1], device=device, samples=128, steps=3, elite=16, seed=seed)

    pending, selections, rows = [], [], []
    try:
        for root in roots:
            seed = root["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != root[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("metadata-only planned root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            tensors = prepare(context)
            pixels, goal, history = tensors
            finite_difference = fd._jacobian_and_fd(model, pixels, goal, history, history_len=3, future_horizon=1,
                probe_dims=8, fd_pairs=8, epsilon=.001, action_lower=lower, action_upper=upper, seed=seed, device=device)
            start_samples = samples
            bank, receipt = words.compile_words(model, pixels, goal, history, lower=lower, upper=upper,
                progress=lambda done, total: print(json.dumps({"seed": seed, "compiled_words": done, "total": total}), flush=True) if done in (8, 16, total) else None)
            gates = {"model_predict_accounting_exact": samples - start_samples == receipt["model_predict_sample_evaluations"],
                "multi_packet_model_partition": receipt["partition_max_abs_error"] <= protocol["partition_atol"],
                "fd_preflight": finite_difference["median_relative_error"] <= protocol["fd_median_relative_error_max"]}
            if not all(gates.values()):
                raise RuntimeError("word compiler implementation preflight failed: " + str(gates))
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            flat, flat_receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=6,
                action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
            cem_start = samples
            cem_full, _, cem_receipt = cem_search(tensors, 6, seed)
            selection = {**root, "receipt": receipt, "fd_preflight": finite_difference, "gates": gates,
                "word_params": bank.tolist(), "flat_params": flat.tolist(), "flat_receipt": flat_receipt,
                "cem_params": cem_full.tolist(), "cem_receipt": cem_receipt,
                "initial_cem_predict_samples": samples - cem_start,
                "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"]}
            selections.append(selection)
            pending.append((root, context, bank, flat, cem_full, selection))
            print(json.dumps({"seed": seed, "sealed_decision": receipt["dispatch"], "gates": gates}), flush=True)
        with paths[2].open("x") as handle:
            handle.write(json.dumps({"protocol_sha256": digest(paths[1]), "rows": selections}, indent=2, sort_keys=True) + "\n")
        selection_sha = digest(paths[2])
        print("All root choices sealed. Physically evaluating EVERY word and both flat controls.", flush=True)
        for root, context, bank, flat, cem_full, selection in pending:
            seed, receipt = root["seed"], selection["receipt"]
            methods, word_results, gates = {}, {}, dict(selection["gates"])
            branches = [("word_" + str(i), bank[i:i + 1], False) for i in range(len(bank))]
            branches += [("flat_svd_gn", flat[:, :4], False), ("cem_mpc", cem_full[:, :4], True)]
            for name, params, is_cem in branches:
                first = raw(params)
                root_frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(root_frame, goal_rgb, context["root_frame"], context["goal"])
                advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=root_frame)
                next_tensors = prepare(advanced)
                pixels, goal, history = next_tensors
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                first_cost = float((emb[0, -1] - goal_latent).square().mean())
                suffix_start = samples
                if is_cem:
                    suffix, _, suffix_receipt = cem_search(next_tensors, 2, seed + 50000)
                else:
                    suffix, suffix_receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                        action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                    if samples - suffix_start != suffix_receipt["model_predict_sample_evaluations"]:
                        raise RuntimeError("observed suffix model-accounting drift")
                suffix_raw = raw(suffix)
                execution._execute_raw_steps(env, suffix_raw)
                physics = env.unwrapped.env.physics
                target = np.asarray(env.unwrapped.env.task.target_qpos)
                physical = {"final_qpos_distance": p0._qpos_distance(np.asarray(physics.data.qpos), target)}
                actions = np.concatenate((first, suffix_raw))
                cell = {"physical": physical, "observed_prefix_rgb_cost": first_cost, "action_witness_raw": actions.tolist(),
                    "prefix_boundaries": boundaries, "execution_identity": identity, "suffix_receipt": suffix_receipt,
                    "suffix_predict_samples": samples - suffix_start, "four_packet_history_updates": len(boundaries) == 4}
                if actions.shape != (30, 2) or np.max(np.abs(actions)) > 1 + 1e-6 or not np.isfinite(actions).all() or not cell["four_packet_history_updates"]:
                    raise RuntimeError("physical action/history contract failed")
                if name.startswith("word_"):
                    word_results[name] = cell
                else:
                    methods[name] = cell
                print(json.dumps({"seed": seed, "physical_branch": name, "final_distance": physical["final_qpos_distance"]}), flush=True)
            decision = receipt["dispatch"]
            for name, index in (("dispatch", decision["selected_program_index"]), ("same_bank_greedy", decision["same_bank_greedy_index"]), ("zero", 0)):
                methods[name] = word_results["word_" + str(index)]
            gates.update({"all_words_physically_retained": len(word_results) == 17,
                "all_history_updates_four_packets": all(cell["four_packet_history_updates"] for cell in list(word_results.values()) + list(methods.values())),
                "true_cem_samples_exact": selection["initial_cem_predict_samples"] + methods["cem_mpc"]["suffix_predict_samples"] == 3072,
                "all_final_distances_finite": all(np.isfinite(cell["physical"]["final_qpos_distance"]) for cell in word_results.values())})
            row = {**root, "decision": decision, "methods": methods, "word_results": word_results, "gates": gates,
                "actual_rgb_non_greedy": methods["dispatch"]["observed_prefix_rgb_cost"] >= methods["zero"]["observed_prefix_rgb_cost"] + 1e-6}
            with (paths[3] / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            if not all(gates.values()):
                raise RuntimeError("physical preflight failed; partial evidence retained")
    finally:
        model.predict = predict
        env.close()
    eligible = [row for row in rows if row["decision"]["reconfiguration_witness_found"]]
    aggregate = {"roots": len(rows), "witnesses": len(eligible), "witness_actual_rgb_non_greedy": sum(row["actual_rgb_non_greedy"] for row in eligible),
        "mean_final_distance": {name: float(np.mean([row["methods"][name]["physical"]["final_qpos_distance"] for row in rows]))
            for name in ("dispatch", "same_bank_greedy", "zero", "flat_svd_gn", "cem_mpc")},
        "paired": {name: runner._paired(rows, "dispatch", name, protocol["bootstrap_seed"]) for name in ("same_bank_greedy", "zero", "flat_svd_gn", "cem_mpc")}}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots_retained": [row["seed"] for row in rows] == seeds,
        "source_episodes_disjoint": not ({row["source_episode"] for row in rows} & excluded),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows), "selection_unchanged": digest(paths[2]) == selection_sha,
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_model_native_action_words.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "selection_sha256": selection_sha,
        "actual_model_predict_samples": samples, "aggregate": aggregate, "rows": rows, "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
