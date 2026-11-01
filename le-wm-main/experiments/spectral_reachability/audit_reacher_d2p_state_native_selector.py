"""All-root/all-word audit of numerical prefix score vs deployment contract.

Rebuild every historical fine8 suffix exactly, then give every word the same
state-native two-chart/CEM optimizer. Preserve exact command witnesses across
the real RGB observation. All model decisions precede terminal outcomes.
Known-root intervention, unequal compute; no P1 admission or theory verdict.
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
from experiments.spectral_reachability import reacher_d2p_state_suffix as native
from experiments.spectral_reachability import reacher_d2p_suffix_continuation as continuation
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference


FAMILIES = ("old_dispatch", "native_dispatch", "greedy", "zero", "cem")


def join_reference(report, prefixes, suffixes):
    required = {"cuda_only", "weights_unchanged", "all_declared_roots", "source_episodes_disjoint", "all_execution_gates",
        "all24_branches_per_root", "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_suffix_multistart_confirmation.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())):
        raise ValueError("complete strict integrity-passing multistart reference required")
    seeds = sorted(report["sealed_protocol"]["root_seeds"])
    rows = sorted(report["rows"], key=lambda r: r["seed"])
    p = {r["seed"]: r for r in prefixes["rows"]}
    s = {(r["seed"], r["family"]): r for r in suffixes["rows"]}
    if (len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds
            or len(p) != len(prefixes["rows"]) or sorted(p) != seeds or len(s) != len(suffixes["rows"])
            or set(s) != {(seed, name) for seed in seeds for name in ("dispatch", "greedy", "zero", "cem")}):
        raise ValueError("complete unique root/family identity joins required")
    for row in rows:
        seed = row["seed"]
        if any(row[k] != p[seed][k] for k in ("source_episode", "source_start_row")):
            raise ValueError("source identity mismatch")
        for name in ("dispatch", "greedy", "zero", "cem"):
            if row["branches"][name] != s[seed, name]:
                raise ValueError("sealed suffix branch mismatch")
    return [(row, p[row["seed"]]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    old_prefix = args.reference.with_suffix(".selection.json")
    old_suffix = args.reference.with_suffix(".suffix.selection.json")
    old_protocol = args.reference.with_suffix(".protocol.json")
    prefixes, suffixes = json.loads(old_prefix.read_text()), json.loads(old_suffix.read_text())
    ref = reference["sealed_protocol"]
    if (digest(old_prefix) != reference["selection_sha256"] or digest(old_suffix) != reference["suffix_selection_sha256"]
            or json.loads(old_protocol.read_text()) != ref or prefixes["protocol_sha256"] != digest(old_protocol)
            or suffixes["protocol_sha256"] != digest(old_protocol) or suffixes["prefix_selection_sha256"] != digest(old_prefix)):
        raise ValueError("historical protocol/prefix/suffix seal changed")
    joined = join_reference(reference, prefixes, suffixes)
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("frozen checkpoint or action statistics changed")
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    modules = {"auditor": Path(__file__), "words": Path(words.__file__), "state_suffix": Path(native.__file__),
        "continuation": Path(continuation.__file__), "multistart": Path(native.historical.__file__), "box": Path(native.box.__file__),
        "qp": Path(native.box.solve_box_quadratic.__code__.co_filename), "p0": Path(p0.__file__), "prefix": Path(p1.__file__),
        "solver": Path(solver.__file__), "rescue": Path(rescue.__file__), "context": Path(execution.__file__),
        "cem": Path(native.cem.__file__), "source_support": Path(support.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "base_selection": Path(words.refinement.__file__),
        "rgb_batch": Path(terminal_rgb_batch.__code__.co_filename)}
    for key, path in modules.items():
        if key != "auditor" and key in ref["implementation_sha256"] and digest(path) != ref["implementation_sha256"][key]:
            raise ValueError("historical implementation changed: " + key)
    output = args.output.resolve()
    paths = {name: output.with_suffix(suffix) for name, suffix in (("protocol", ".protocol.json"), ("prefix", ".selection.json"),
        ("suffix", ".suffix.selection.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference), "root_seeds": ref["root_seeds"],
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "target_delta": ref["target_delta"], "all17_words_rebuilt": True, "non_greedy_margin": 1e-6,
        "optimizer": native.POLICY, "extra_iterations": [8, 32], "damping": .01,
        "cem": {"samples": 128, "steps": 3, "elite": 16, "horizon": 2, "seed_offset": 70000, "common_random_numbers_across_words": True},
        "physical_raw_steps": 30, "prefix_families": list(FAMILIES), "physical_branches_per_root": 16,
        "all_prefix_choices_sealed": True, "all_suffix_choices_sealed_before_terminal_execution": True,
        "physical_labels_used_for_planning": False, "partition_atol": 2e-6,
        "historical_native_commands_and_costs_atol": 0., "historical_physical_replay_atol": 0.,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root score/execution contract intervention; unequal compute, not fresh efficacy, P1 admission or theory verdict."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (key + ".py"))
    print("All8 roots/all17 words: native numerical score, exact command carry, matched prefix controls. Historical optimizer/physical replay must be exact; CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != ref["source_sha256"] or support._sha256(dataset) != source_sha:
        raise ValueError("official source SHA changed")
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
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

    def raw_params(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def raw_commands(commands):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * commands)[0].detach().cpu().numpy()
        return p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)

    choices, pending, suffix_choices, terminal_jobs, rows = [], [], [], [], []
    try:
        for old, choice in joined:
            seed = old["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if (any(context[k] != old[k] for k in ("source_episode", "source_start_row"))
                    or any(context[k] != choice[k] for k in ("root_rgb_sha256", "goal_rgb_sha256"))):
                raise RuntimeError("historical source/RGB root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            # Exact real-observation equivalence of the new cached-state core.
            frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
            first = np.asarray(old["methods"]["dispatch_incumbent"]["action_witness_raw"], np.float32)[:20]
            advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=frame)
            pixels, goal, history = prepare(advanced)
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            old_index = choice["receipt"]["dispatch"]["selected_program_index"]
            imagined_old = torch.tensor(choice["planned_tails"][str(old_index)], device=device, dtype=history.dtype)
            local, local_receipt = continuation.compile_continuation(model, goal_latent, emb, history,
                planned_suffix=imagined_old, action_lower=lower, action_upper=upper)
            incumbent = local[local_receipt["selected"]]
            start = samples
            native_commands, native_receipt = native.compile_state_candidates(model, goal_latent, emb, history,
                reference_params=incumbent, action_lower=lower, action_upper=upper, seed=seed + 70000)
            old_branch = old["branches"]["dispatch"]
            core_gates = {"local_receipt_exact": local_receipt == old_branch["local_receipt"],
                "prefix_history_exact": boundaries == old_branch["prefix_boundaries"],
                "native_core_model_costs_exact": native_receipt["model_costs"] == old_branch["receipt"]["model_costs"],
                "native_core_receipts_exact": native_receipt["receipts"] == old_branch["receipt"]["receipts"],
                "native_core_commands_exact": all(np.array_equal(raw_commands(value), np.asarray(old_branch["suffix_actions_raw"][name], np.float32)) for name, value in native_commands.items()),
                "native_core_accounting": samples - start == native_receipt["model_predict_sample_evaluations"]}
            write_new(paths["roots"] / f"core_{seed}.json", {"seed": seed, "gates": core_gates})
            if not all(core_gates.values()):
                raise RuntimeError("native observed-state equivalence failed: " + str(core_gates))
            pixels, goal, history = prepare(context)
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)

            def terminal(flat):
                return p0._planning_coordinate(model, p1._terminal_from_state(model, emb, history,
                    flat.reshape(1, 1, history.shape[-1]), action_lower=lower, action_upper=upper)[0], role="predicted")

            origin = torch.zeros(history.shape[-1], device=device, dtype=history.dtype, requires_grad=True)
            response = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
            bank, word_support = words.build_words(response.detach())
            if (not torch.equal(bank, torch.tensor(choice["word_params"], device=device, dtype=bank.dtype))
                    or json.loads(json.dumps(word_support)) != choice["receipt"]["word_support"]):
                raise RuntimeError("historical action word bank changed")
            score_rows, word_records, tails = [], [], {}
            for index in range(17):
                with torch.no_grad():
                    immediate = float(p0._cost_fn(model, pixels, goal, history, bank[index:index + 1], action_lower=lower, action_upper=upper)[0])
                next_emb, next_actions = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
                suffix, original = solver.compile_snapshots(model, goal_latent, next_emb, next_actions, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                old_score = choice["receipt"]["proposal_rows"][index]
                if immediate != old_score["immediate_goal_cost"] or original != old_score["suffix_receipt"]:
                    raise RuntimeError("historical per-word score/suffix did not reproduce exactly")
                start = samples
                variants, receipt = native.compile_state_candidates(model, goal_latent, next_emb, next_actions,
                    reference_params=suffix, action_lower=lower, action_upper=upper, seed=seed + 70000)
                if samples - start != receipt["model_predict_sample_evaluations"]:
                    raise RuntimeError("imagined native suffix accounting drift")
                planned = variants[receipt["selected"]]
                with torch.no_grad():
                    whole = native.box.terminal_commands(model, emb, history, torch.cat((bank[index:index + 1].tanh(), planned), dim=1), action_lower=lower, action_upper=upper)
                    split = native.box.terminal_commands(model, next_emb, next_actions, planned, action_lower=lower, action_upper=upper)
                    partition = float((whole - split).abs().max())
                if partition > 2e-6:
                    raise RuntimeError("new complete command witness is not partition-consistent")
                score_rows.append({"proposal_index": index, "immediate_goal_cost": immediate, "refined_goal_cost": receipt["model_costs"][receipt["selected"]]})
                tails[index] = planned
                word_records.append({"index": index, "original_fine8_cost": old_score["refined_goal_cost"],
                    "native_receipt": receipt, "planned_commands": planned.tolist(), "partition_max_abs_error": partition})
                if index % 4 == 0 or index == 16:
                    print(json.dumps({"seed": seed, "word_completed": index + 1, "of": 17}), flush=True)
            decision = words.refinement.select_refined_bank(score_rows, non_greedy_margin=1e-6)
            record = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "historical_decision": choice["receipt"]["dispatch"], "native_decision": decision,
                "score_rows": score_rows, "word_records": word_records, "gates": core_gates}
            write_new(paths["roots"] / f"scores_{seed}.json", record)
            choices.append(record)
            pending.append((old, choice, context, bank, tails, record))
            print(json.dumps({"seed": seed, "old_decision": record["historical_decision"], "native_decision": decision}), flush=True)
        write_new(paths["prefix"], {"protocol_sha256": digest(paths["protocol"]), "rows": choices})
        prefix_sha = digest(paths["prefix"])
        print("All136 imagined suffixes and every native prefix choice sealed; observe matched prefixes.", flush=True)
        for old, old_choice, context, bank, tails, choice in pending:
            seed = old["seed"]
            indices = {"old_dispatch": choice["historical_decision"]["selected_program_index"],
                "native_dispatch": choice["native_decision"]["selected_program_index"],
                "greedy": choice["native_decision"]["same_bank_greedy_index"], "zero": 0}
            branches = [(name, bank[index:index + 1], tails[index]) for name, index in indices.items()]
            full_cem = torch.tensor(old_choice["cem_params"], device=device, dtype=bank.dtype)
            branches.append(("cem", full_cem[:, :4], full_cem[:, 4:].tanh()))
            for family, prefix, planned in branches:
                frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                first = raw_params(prefix)
                advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=frame)
                pixels, goal, history = prepare(advanced)
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                start = samples
                cold, cold_receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                variants, receipt = native.compile_state_candidates(model, goal_latent, emb, history, reference_params=cold,
                    planned_commands=planned, action_lower=lower, action_upper=upper, seed=seed + 70000)
                if samples - start != cold_receipt["model_predict_sample_evaluations"] + receipt["model_predict_sample_evaluations"]:
                    raise RuntimeError("observed native suffix accounting drift")
                suffix_raw = {"cold": raw_params(cold), "carry": raw_commands(planned), "native": raw_commands(variants[receipt["selected"]])}
                if family == "old_dispatch":
                    suffix_raw["historical"] = np.asarray(old["methods"]["dispatch_model_best"]["action_witness_raw"], np.float32)[20:]
                record = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                    "family": family, "prefix_index": indices.get(family), "prefix_actions_raw": first.tolist(),
                    "prefix_boundaries": boundaries, "execution_identity": identity,
                    "prefix_rgb_cost": float((emb[0, -1] - goal_latent).square().mean()),
                    "planned_commands": planned.tolist(), "native_commands": variants[receipt["selected"]].tolist(),
                    "receipt": receipt, "cold_receipt": cold_receipt,
                    "suffix_actions_raw": {name: value.tolist() for name, value in suffix_raw.items()}}
                suffix_choices.append(record)
                terminal_jobs.append((old, advanced, suffix_raw, record))
                print(json.dumps({"seed": seed, "family": family, "selected": receipt["selected"], "model_costs": receipt["model_costs"]}), flush=True)
        write_new(paths["suffix"], {"protocol_sha256": digest(paths["protocol"]), "prefix_selection_sha256": prefix_sha, "rows": suffix_choices})
        suffix_sha = digest(paths["suffix"])
        print("All40 observed native suffix choices sealed; begin128 physical terminal branches.", flush=True)
        for old, _ in joined:
            seed, methods, branches = old["seed"], {}, {}
            for job_old, advanced, suffix_raw, record in terminal_jobs:
                if job_old["seed"] != seed:
                    continue
                family = record["family"]
                for name, actions in suffix_raw.items():
                    execution.restore_execution_snapshot(env, seed=seed, snapshot=advanced["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    execution._execute_raw_steps(env, actions)
                    physics = env.unwrapped.env.physics
                    distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                    prepared = processor._prepare_info(terminal_rgb_batch(env.render(), advanced["goal"], mean.size))
                    rgb_cost = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                    complete = np.concatenate((np.asarray(record["prefix_actions_raw"], np.float32), actions))
                    if complete.shape != (30, 2) or not np.isfinite(complete).all() or np.abs(complete).max() > 1 + 1e-6 or not np.isfinite([distance, rgb_cost]).all():
                        raise RuntimeError("bounded physical action/metric contract failed")
                    methods[family + "_" + name] = {"action_witness_raw": complete.tolist(), "physical": {"final_qpos_distance": distance, "final_rgb_cost": rgb_cost}}
                branches[family] = record
            previous = old["methods"]["dispatch_model_best"]
            replay = methods["old_dispatch_historical"]
            gates = {"historical_complete_actions_exact": replay["action_witness_raw"] == previous["action_witness_raw"],
                "historical_distance_exact": replay["physical"]["final_qpos_distance"] == previous["physical"]["final_qpos_distance"],
                "historical_rgb_cost_exact": replay["physical"]["final_rgb_cost"] == previous["physical"]["final_rgb_cost"],
                "all16_physical_branches": len(methods) == 16 and set(branches) == set(FAMILIES)}
            row = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "methods": methods, "branches": branches, "gates": gates}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "distances": {k: v["physical"]["final_qpos_distance"] for k, v in methods.items() if k.endswith(("_native", "_historical"))}, "gates": gates}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("historical complete physical replay failed")
    finally:
        model.predict = predict
        env.close()
    metrics = {name: [r["methods"][name]["physical"]["final_qpos_distance"] for r in rows] for name in rows[0]["methods"]}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots": [r["seed"] for r in rows] == sorted(ref["root_seeds"]),
        "all17_words_every_root": all(len(r["word_records"]) == 17 for r in choices),
        "all_core_equivalence_gates": all(all(r["gates"].values()) for r in choices),
        "all_physical_gates": all(all(r["gates"].values()) for r in rows),
        "prefix_seal_unchanged": digest(paths["prefix"]) == prefix_sha, "suffix_seal_unchanged": digest(paths["suffix"]) == suffix_sha,
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    aggregate = {"roots": 8, "imagined_words": 136, "physical_branches": 128,
        "mean_distance": {k: float(np.mean(v)) for k, v in metrics.items()},
        "historical_witnesses": sum(c["historical_decision"]["reconfiguration_witness_found"] for c in choices),
        "native_witnesses": sum(c["native_decision"]["reconfiguration_witness_found"] for c in choices),
        "prefix_choices_changed": sum(c["historical_decision"]["selected_program_index"] != c["native_decision"]["selected_program_index"] for c in choices),
        "comparisons": {name: paired_difference(metrics["native_dispatch_native"], metrics[name]) for name in
            ("old_dispatch_historical", "old_dispatch_native", "greedy_native", "zero_native", "cem_native")}}
    write_new(output, {"schema": "reacher_d2p_state_native_selector_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": prefix_sha, "suffix_selection_sha256": suffix_sha,
        "gates": gates, "aggregate": aggregate, "rows": rows, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
