"""Prospective episode-disjoint confirmation of planned-tail preservation.

Every method sees the same root, 20-step prefix, observation and 10-step tail.
Compare cold/continued suffixes on dispatch, greedy, neutral and flat SVD/GN;
also retain the initial CEM tail as an observed-model incumbent for CEM.
This pilot does not change the P1 prefix-ranking gate or claim equal compute.
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
from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_suffix_continuation as continuation
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import preflight_reacher_d2p_p0 as fd
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202643000)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    if args.roots < 8:
        raise ValueError("at least eight predeclared roots required")
    output = args.output.resolve()
    paths = {name: output.with_suffix(suffix) for name, suffix in
        (("protocol", ".protocol.json"), ("selection", ".selection.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(path.exists() for path in [output, *paths.values()]):
        raise FileExistsError("new immutable outputs required")
    seeds = list(range(args.seed, args.seed + args.roots))
    excluded, bindings = bind_exclusions(args.disjoint_reference, seeds)
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    modules = {"auditor": Path(__file__), "words": Path(words.__file__), "continuation": Path(continuation.__file__),
        "p0": Path(p0.__file__), "prefix": Path(p1.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "cem": Path(cem.__file__), "fd": Path(fd.__file__), "source_support": Path(support.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "base_selection": Path(words.refinement.__file__)}
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": digest(args.policy / "weights.pt"), "adapter_sha256": digest(args.adapter_artifact), "source_sha256": source_sha,
        "target_delta": [.65, -.65], "proposals": 17, "prefix_packets": 4, "suffix_packets": 2, "action_block": 5,
        "physical_raw_steps": 30, "damping": .01, "iterations": 8, "compiler": words.POLICY,
        "runtime": "preserve_planned_suffix_cold_warm_fine8_incumbent_v1", "prefix_selector_changed": False,
        "cem": {"horizons": [6, 2], "samples": 128, "steps": 3, "elite": 16, "suffix_seed_offset": 50000,
            "incumbent": "minimum singleton observed-model cost: carried initial CEM tail or newly searched CEM tail"},
        "all_choices_sealed_before_prefix_execution": True, "all_suffix_choices_sealed_before_terminal_execution": True,
        "physical_labels_used_for_planning": False, "bootstrap_seed": args.seed + 911,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Prospective source-episode-disjoint repair confirmation; equal physical duration but unequal planning compute; not formal P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (key + ".py"))
    print("Prospective all8 root confirmation: preserve planned tails, cold/continuation on all four prefix controls, CEM tail incumbent; CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    if support._sha256(dataset) != source_sha:
        raise ValueError("official source SHA changed")
    device = torch.device("cuda")
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
                raise RuntimeError("metadata root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            tensors = prepare(context)
            pixels, goal, history = tensors
            finite_difference = fd._jacobian_and_fd(model, pixels, goal, history, history_len=3, future_horizon=1,
                probe_dims=8, fd_pairs=8, epsilon=.001, action_lower=lower, action_upper=upper, seed=seed, device=device)
            start = samples
            bank, receipt = words.compile_words(model, pixels, goal, history, lower=lower, upper=upper)
            compiler_samples = samples - start
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            # Leave the immutable historical compiler and its default receipts
            # unchanged. Reconstruct the three required tails deterministically
            # and count this explicit diagnostic overhead, rather than silently
            # pretending discarded params were stored by the old compiler.
            required = {0, receipt["dispatch"]["selected_program_index"], receipt["dispatch"]["same_bank_greedy_index"]}
            planned_tails = {}
            reconstruct_start = samples
            for index in sorted(required):
                next_emb, next_actions = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
                tail, tail_receipt = solver.compile_snapshots(model, goal_latent, next_emb, next_actions, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                if tail_receipt != receipt["proposal_rows"][index]["suffix_receipt"]:
                    raise RuntimeError("imagined continuation tail cannot be reconstructed exactly")
                planned_tails[str(index)] = tail.tolist()
            reconstruction_samples = samples - reconstruct_start
            flat, flat_receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=6,
                action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
            cem_start = samples
            cem_full, _, cem_receipt = cem_search(tensors, 6, seed)
            initial_cem_samples = samples - cem_start
            gates = {"fd_preflight": finite_difference["median_relative_error"] <= .1,
                "compiler_accounting_exact": compiler_samples == receipt["model_predict_sample_evaluations"],
                "model_partition": receipt["partition_max_abs_error"] <= 2e-6,
                "required_tails_exactly_reconstructed": set(planned_tails) == {str(index) for index in required}}
            if not all(gates.values()):
                raise RuntimeError("compiler preflight failed: " + str(gates))
            choice = {**root, "receipt": receipt, "word_params": bank.tolist(), "flat_params": flat.tolist(), "flat_receipt": flat_receipt,
                "cem_params": cem_full.tolist(), "cem_receipt": cem_receipt, "initial_cem_predict_samples": initial_cem_samples,
                "planned_word_suffixes": planned_tails, "carry_tail_reconstruction_predict_samples": reconstruction_samples,
                "gates": gates, "fd_preflight": finite_difference, "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"]}
            selections.append(choice)
            pending.append((root, context, bank, flat, cem_full, choice))
            print(json.dumps({"seed": seed, "sealed_decision": receipt["dispatch"], "gates": gates}), flush=True)
        write_new(paths["selection"], {"protocol_sha256": digest(paths["protocol"]), "rows": selections})
        selection_sha = digest(paths["selection"])
        print("All prospective prefix decisions and imagined tails sealed. Begin matched-prefix terminal measurements.", flush=True)
        for root, context, bank, flat, cem_full, choice in pending:
            seed, methods, branch_records = root["seed"], {}, {}
            decision, gates = choice["receipt"]["dispatch"], dict(choice["gates"])
            branches = []
            for name, index in (("dispatch", decision["selected_program_index"]), ("greedy", decision["same_bank_greedy_index"]), ("zero", 0)):
                tail = torch.tensor(choice["planned_word_suffixes"][str(index)], device=device, dtype=bank.dtype)
                branches.append((name, bank[index:index + 1], tail))
            branches += [("flat", flat[:, :4], flat[:, 4:]), ("cem", cem_full[:, :4], cem_full[:, 4:])]
            for name, prefix, planned in branches:
                root_frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(root_frame, goal_rgb, context["root_frame"], context["goal"])
                first = raw(prefix)
                advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=root_frame)
                tensors = prepare(advanced)
                pixels, goal, history = tensors
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                prefix_cost = float((emb[0, -1] - goal_latent).square().mean())
                start = samples
                if name == "cem":
                    replan, _, replan_receipt = cem_search(tensors, 2, seed + 50000)
                    search_samples = samples - start
                    with torch.no_grad():
                        costs = {key: float(p0._cost_fn(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)[0])
                            for key, params in (("carry", planned), ("replan", replan))}
                    selected = min(("replan", "carry"), key=lambda key: (costs[key], key != "replan"))
                    suffixes = {"cem_replan": replan, "cem_incumbent": planned if selected == "carry" else replan}
                    suffix_receipt = {"selected": selected, "model_costs": costs, "replan_receipt": replan_receipt,
                        "physical_labels_used": False, "model_predict_sample_evaluations": samples - start}
                    gates["cem_search_samples_exact"] = search_samples + choice["initial_cem_predict_samples"] == 3072
                else:
                    variants, suffix_receipt = continuation.compile_continuation(model, goal_latent, emb, history,
                        planned_suffix=planned, action_lower=lower, action_upper=upper)
                    suffixes = {name + "_cold": variants["cold"], name + "_incumbent": variants[suffix_receipt["selected"]]}
                    gates[name + "_continuation_accounting"] = samples - start == suffix_receipt["model_predict_sample_evaluations"]
                suffix_path = paths["roots"] / f"suffix_{seed}_{name}.selection.json"
                write_new(suffix_path, {"seed": seed, "name": name, "prefix_boundaries": boundaries, "receipt": suffix_receipt,
                    "suffix_params": {key: params.tolist() for key, params in suffixes.items()}})
                suffix_sha = digest(suffix_path)
                branch_records[name] = {"receipt": suffix_receipt, "suffix_selection_sha256": suffix_sha}
                for key, params in suffixes.items():
                    execution.restore_execution_snapshot(env, seed=seed, snapshot=advanced["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    suffix_raw = raw(params)
                    execution._execute_raw_steps(env, suffix_raw)
                    physics = env.unwrapped.env.physics
                    distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                    actions = np.concatenate((first, suffix_raw))
                    methods[key] = {"physical": {"final_qpos_distance": distance}, "observed_prefix_rgb_cost": prefix_cost,
                        "execution_identity": identity, "action_witness_raw": actions.tolist(), "prefix_boundaries": boundaries}
                    if actions.shape != (30, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
                        raise RuntimeError("bounded physical action contract failed")
                gates[name + "_suffix_seal_unchanged"] = digest(suffix_path) == suffix_sha
            gates.update({"all10_methods": len(methods) == 10,
                "all_distances_finite": all(np.isfinite(item["physical"]["final_qpos_distance"]) for item in methods.values()),
                "prefix_actions_shared": all(methods[name + "_cold"]["action_witness_raw"][:20] == methods[name + "_incumbent"]["action_witness_raw"][:20]
                    for name in ("dispatch", "greedy", "zero", "flat"))})
            row = {**root, "decision": decision, "methods": methods, "branches": branch_records, "gates": gates,
                "actual_rgb_non_greedy": methods["dispatch_incumbent"]["observed_prefix_rgb_cost"] >= methods["zero_incumbent"]["observed_prefix_rgb_cost"] + 1e-6}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "distances": {key: item["physical"]["final_qpos_distance"] for key, item in methods.items()}, "gates": gates}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("prospective physical integrity gate failed")
    finally:
        model.predict = predict
        env.close()
    names = sorted(rows[0]["methods"])
    eligible = [row for row in rows if row["decision"]["reconfiguration_witness_found"]]
    aggregate = {"roots": len(rows), "methods": len(names), "physical_branches": len(rows) * len(names),
        "witnesses": len(eligible), "witness_actual_rgb_non_greedy": sum(row["actual_rgb_non_greedy"] for row in eligible),
        "mean_final_distance": {name: float(np.mean([row["methods"][name]["physical"]["final_qpos_distance"] for row in rows])) for name in names},
        "paired": {name: runner._paired(rows, "dispatch_incumbent", name, protocol["bootstrap_seed"]) for name in names if name != "dispatch_incumbent"}}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots": [row["seed"] for row in rows] == seeds,
        "source_episodes_disjoint": not ({row["source_episode"] for row in rows} & excluded),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows), "selection_unchanged": digest(paths["selection"]) == selection_sha,
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    write_new(output, {"schema": "reacher_d2p_suffix_continuation_confirmation.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": selection_sha,
        "gates": gates, "aggregate": aggregate, "rows": rows, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
