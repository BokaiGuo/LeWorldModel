"""Fresh all-root confirmation of two-chart/CEM numerical suffix repairs.

Four prefix controls: unchanged word dispatch, same-bank greedy, neutral,
and full6-token CEM. Every suffix candidate is sealed before any terminal
measurement. Model-only selection; no P1 admission or compute-match claim.
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
from experiments.spectral_reachability import reacher_d2p_suffix_multistart as multistart
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import preflight_reacher_d2p_p0 as fd
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--adapter-artifact", required=True, type=Path)
    parser.add_argument("--disjoint-reference", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=202644000)
    parser.add_argument("--roots", type=int, default=8)
    args = parser.parse_args()
    if args.roots != 8:
        raise ValueError("all eight predeclared roots required")
    output = args.output.resolve()
    paths = {name: output.with_suffix(suffix) for name, suffix in
        (("protocol", ".protocol.json"), ("selection", ".selection.json"), ("suffix", ".suffix.selection.json"),
         ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    seeds = list(range(args.seed, args.seed + args.roots))
    for path in args.disjoint_reference:
        if not all(v is True for v in json.loads(path.read_text()).get("gates", {}).values()):
            raise ValueError("strict exclusion gates required")
    excluded, bindings = bind_exclusions(args.disjoint_reference, seeds)
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    modules = {"auditor": Path(__file__), "words": Path(words.__file__), "continuation": Path(continuation.__file__),
        "multistart": Path(multistart.__file__), "box": Path(multistart.box.__file__),
        "qp": Path(multistart.box.solve_box_quadratic.__code__.co_filename),
        "p0": Path(p0.__file__), "prefix": Path(p1.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "cem": Path(cem.__file__), "fd": Path(fd.__file__), "source_support": Path(support.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "base_selection": Path(words.refinement.__file__),
        "rgb_batch": Path(terminal_rgb_batch.__code__.co_filename)}
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": digest(args.policy / "weights.pt"), "adapter_sha256": digest(args.adapter_artifact), "source_sha256": source_sha,
        "target_delta": [.65, -.65], "physical_raw_steps": 30, "prefix_families": ["dispatch", "greedy", "zero", "cem"],
        "suffix_candidates": list(multistart.NAMES), "extra_iterations": [8, 32], "damping": .01,
        "prefix_selector_changed": False, "all_prefix_choices_sealed": True, "all_suffix_choices_sealed_before_terminal_execution": True,
        "physical_labels_used_for_planning": False, "cem": {"samples": 128, "steps": 3, "elite": 16,
            "initial_horizon": 6, "suffix_horizon": 2, "suffix_seed_offset": 70000},
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Prospective numerical suffix repair confirmation, unequal compute, not P1 admission or mechanism efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (key + ".py"))
    print("Fresh all8 numerical repair confirmation; dispatch/greedy/neutral/full-CEM prefixes; seal ALL suffixes before terminal outcomes; CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    if support._sha256(dataset) != source_sha:
        raise ValueError("complete official source SHA changed")
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before, predict, samples = shared._parameter_hash(model), model.predict, 0

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
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def raw_commands(commands):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * commands)[0].detach().cpu().numpy()
        return p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)

    pending, choices, suffix_choices, rows = [], [], [], []
    try:
        for root in roots:
            seed = root["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != root[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("source root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = prepare(context)
            finite = fd._jacobian_and_fd(model, pixels, goal, history, history_len=3, future_horizon=1,
                probe_dims=8, fd_pairs=8, epsilon=.001, action_lower=lower, action_upper=upper, seed=seed, device=device)
            start = samples
            bank, receipt = words.compile_words(model, pixels, goal, history, lower=lower, upper=upper)
            gates = {"fd_preflight": finite["median_relative_error"] <= .1,
                "compiler_accounting_exact": samples - start == receipt["model_predict_sample_evaluations"],
                "model_partition": receipt["partition_max_abs_error"] <= 2e-6}
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            tails = {}
            for index in sorted({0, receipt["dispatch"]["selected_program_index"], receipt["dispatch"]["same_bank_greedy_index"]}):
                next_emb, next_actions = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
                tail, reconstructed = solver.compile_snapshots(model, goal_latent, next_emb, next_actions, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                if reconstructed != receipt["proposal_rows"][index]["suffix_receipt"]:
                    raise RuntimeError("imagined suffix reconstruction changed")
                tails[str(index)] = tail.tolist()
            full_cem, _, cem_receipt = cem.search_with_incumbent(lambda params: p0._cost_fn(model, pixels, goal, history, params,
                action_lower=lower, action_upper=upper), future_horizon=6, action_dim=history.shape[-1], device=device,
                samples=128, steps=3, elite=16, seed=seed)
            if not all(gates.values()):
                raise RuntimeError("prefix preflight failed: " + str(gates))
            choice = {**root, "receipt": receipt, "word_params": bank.tolist(), "planned_tails": tails,
                "cem_params": full_cem.tolist(), "cem_receipt": cem_receipt, "gates": gates,
                "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"]}
            choices.append(choice)
            pending.append((root, context, bank, full_cem, choice))
            print(json.dumps({"seed": seed, "decision": receipt["dispatch"], "gates": gates}), flush=True)
        write_new(paths["selection"], {"protocol_sha256": digest(paths["protocol"]), "rows": choices})
        selection_sha = digest(paths["selection"])
        terminal_jobs = []
        for root, context, bank, full_cem, choice in pending:
            seed, decision = root["seed"], choice["receipt"]["dispatch"]
            branches = [(name, bank[index:index + 1], torch.tensor(choice["planned_tails"][str(index)], device=device, dtype=bank.dtype))
                for name, index in (("dispatch", decision["selected_program_index"]), ("greedy", decision["same_bank_greedy_index"]), ("zero", 0))]
            branches.append(("cem", full_cem[:, :4], full_cem[:, 4:]))
            for family, prefix, tail in branches:
                frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                first = raw(prefix)
                advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=frame)
                pixels, goal, history = prepare(advanced)
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                start = samples
                local, local_receipt = continuation.compile_continuation(model, goal_latent, emb, history,
                    planned_suffix=tail, action_lower=lower, action_upper=upper)
                incumbent = local[local_receipt["selected"]]
                commands, receipt = multistart.compile_candidates(model, pixels, goal, history, incumbent=incumbent,
                    action_lower=lower, action_upper=upper, seed=seed + 70000)
                actions = {name: raw_commands(value) for name, value in commands.items()}
                gates = {**choice["gates"], "suffix_accounting_exact": samples - start ==
                    local_receipt["model_predict_sample_evaluations"] + receipt["model_predict_sample_evaluations"],
                    "incumbent_raw_exact": np.array_equal(actions["incumbent"], raw(incumbent)),
                    "no_model_ascent": receipt["model_cost_not_above_incumbent"] is True}
                if not all(gates.values()):
                    raise RuntimeError("suffix integrity failed: " + str(gates))
                record = {**root, "family": family, "prefix_actions_raw": first.tolist(), "prefix_boundaries": boundaries,
                    "execution_identity": identity, "prefix_rgb_cost": float((emb[0, -1] - goal_latent).square().mean()),
                    "suffix_actions_raw": {name: value.tolist() for name, value in actions.items()},
                    "local_receipt": local_receipt, "receipt": receipt, "gates": gates}
                suffix_choices.append(record)
                terminal_jobs.append((root, advanced, actions, record))
                print(json.dumps({"seed": seed, "family": family, "selected": receipt["selected"], "costs": receipt["model_costs"], "gates": gates}), flush=True)
        write_new(paths["suffix"], {"protocol_sha256": digest(paths["protocol"]), "prefix_selection_sha256": selection_sha, "rows": suffix_choices})
        suffix_sha = digest(paths["suffix"])
        print("ALL32 observed suffix comparisons sealed; begin192 physical terminal branches.", flush=True)
        for root in roots:
            methods, branches, gates = {}, {}, {}
            for job_root, advanced, actions, record in terminal_jobs:
                if job_root["seed"] != root["seed"]:
                    continue
                family = record["family"]
                for name, suffix in actions.items():
                    execution.restore_execution_snapshot(env, seed=root["seed"], snapshot=advanced["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    execution._execute_raw_steps(env, suffix)
                    physics = env.unwrapped.env.physics
                    distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                    prepared = processor._prepare_info(terminal_rgb_batch(env.render(), advanced["goal"], mean.size))
                    rgb_cost = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                    complete = np.concatenate((np.asarray(record["prefix_actions_raw"], np.float32), suffix))
                    if complete.shape != (30, 2) or not np.isfinite(complete).all() or np.abs(complete).max() > 1 + 1e-6 or not np.isfinite([distance, rgb_cost]).all():
                        raise RuntimeError("physical witness/metric contract failed")
                    methods[family + "_" + name] = {"physical": {"final_qpos_distance": distance, "final_rgb_cost": rgb_cost}, "action_witness_raw": complete.tolist()}
                selected = record["receipt"]["selected"]
                methods[family + "_model_best"] = methods[family + "_" + selected]
                branches[family] = record
                gates[family + "_integrity"] = all(record["gates"].values())
            row = {**root, "methods": methods, "branches": branches, "gates": gates}
            write_new(paths["roots"] / f"root_{root['seed']}.json", row)
            rows.append(row)
            print(json.dumps({"seed": root["seed"], "distances": {name: value["physical"]["final_qpos_distance"] for name, value in methods.items() if name.endswith(("incumbent", "model_best"))}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    names = sorted(rows[0]["methods"])
    metrics = {name: [r["methods"][name]["physical"]["final_qpos_distance"] for r in rows] for name in names}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots": [r["seed"] for r in rows] == seeds,
        "source_episodes_disjoint": not ({r["source_episode"] for r in rows} & excluded),
        "all_execution_gates": all(all(r["gates"].values()) for r in rows),
        "all24_branches_per_root": all(len(r["methods"]) == 28 and len(r["branches"]) == 4 for r in rows),
        "prefix_seal_unchanged": digest(paths["selection"]) == selection_sha,
        "suffix_seal_unchanged": digest(paths["suffix"]) == suffix_sha,
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    aggregate = {"roots": len(rows), "physical_branches": 24 * len(rows), "mean_distance": {name: float(np.mean(values)) for name, values in metrics.items()},
        "within_prefix_repairs": {family: paired_difference(metrics[family + "_model_best"], metrics[family + "_incumbent"]) for family in protocol["prefix_families"]},
        "dispatch_comparisons": {family: paired_difference(metrics["dispatch_model_best"], metrics[family + "_model_best"]) for family in ("greedy", "zero", "cem")}}
    write_new(output, {"schema": "reacher_d2p_suffix_multistart_confirmation.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": selection_sha, "suffix_selection_sha256": suffix_sha,
        "gates": gates, "aggregate": aggregate, "rows": rows, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
