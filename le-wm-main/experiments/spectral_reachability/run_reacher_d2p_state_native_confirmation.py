"""Source-episode-disjoint confirmation of imagined/deployed suffix alignment.

Every original word score is replayed under the frozen implementation. All
prefix choices and all observed suffix choices are sealed before measuring
any terminal outcome. Neither physical labels nor outcomes select actions.
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
from experiments.spectral_reachability import reacher_d2p_state_native_words as compiler
from experiments.spectral_reachability import reacher_d2p_suffix_continuation as continuation
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability import preflight_reacher_d2p_p0 as fd
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.audit_reacher_d2p_state_native_selector import FAMILIES
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import reacher_d2p_feasible_goal as feasible_goals


def validate_alignment(report):
    required = {"cuda_only", "weights_unchanged", "all_declared_roots", "all17_words_every_root", "all_core_equivalence_gates",
        "all_physical_gates", "prefix_seal_unchanged", "suffix_seal_unchanged", "reference_unchanged", "implementation_unchanged"}
    if (report.get("schema") != "reacher_d2p_state_native_selector_audit.v1" or report.get("status") != "completed"
            or not required.issubset(report.get("gates", {})) or not all(v is True for v in report["gates"].values())
            or len(report.get("rows", [])) != 8):
        raise ValueError("completed all8 numerical-contract audit required before confirmation")
    seeds = report["sealed_protocol"]["root_seeds"]
    if len(set(seeds)) != 8 or sorted(r["seed"] for r in report["rows"]) != sorted(seeds):
        raise ValueError("complete unique alignment root identities required")
    return report["sealed_protocol"]


def load_alignment(path):
    from experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector import summarize

    report = json.loads(path.read_text())
    protocol = validate_alignment(report)
    p, s, t = (path.with_suffix(ext) for ext in (".selection.json", ".suffix.selection.json", ".protocol.json"))
    prefix, suffix = json.loads(p.read_text()), json.loads(s.read_text())
    if (digest(p) != report["selection_sha256"] or digest(s) != report["suffix_selection_sha256"]
            or json.loads(t.read_text()) != protocol or prefix["protocol_sha256"] != digest(t)
            or suffix["protocol_sha256"] != digest(t) or suffix["prefix_selection_sha256"] != digest(p)):
        raise ValueError("alignment protocol or decision seals changed")
    summarize(report, prefix, suffix)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--alignment-reference", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, required=True, action="append")
    parser.add_argument("--seed", type=int, default=202646000)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--goal-contract", choices=goals.GOAL_CONTRACTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.roots != 8:
        raise ValueError("all eight predeclared roots required")
    alignment = load_alignment(args.alignment_reference)
    output = args.output.resolve()
    paths = {name: output.with_suffix(suffix) for name, suffix in (("protocol", ".protocol.json"), ("prefix", ".selection.json"),
        ("suffix", ".suffix.selection.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    seeds = list(range(args.seed, args.seed + 8))
    for path in args.disjoint_reference:
        if not all(v is True for v in json.loads(path.read_text()).get("gates", {}).values()):
            raise ValueError("strict exclusion references required")
    excluded, bindings = bind_exclusions(args.disjoint_reference, seeds)
    if not {r["source_episode"] for r in json.loads(args.alignment_reference.read_text())["rows"]}.issubset(excluded):
        raise ValueError("alignment roots must be excluded from confirmation")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    roots = planned_source_roots(dataset, seeds, excluded)
    words, native = compiler.words, compiler.native
    p0, p1, solver, rescue, cem = words.p0, words.p1, words.solver, words.rescue, native.cem
    modules = {"auditor": Path(__file__), "compiler": Path(compiler.__file__), "words": Path(words.__file__),
        "state_suffix": Path(native.__file__), "continuation": Path(continuation.__file__), "multistart": Path(native.historical.__file__),
        "box": Path(native.box.__file__), "qp": Path(native.box.solve_box_quadratic.__code__.co_filename),
        "p0": Path(p0.__file__), "prefix": Path(p1.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "cem": Path(cem.__file__), "source_support": Path(support.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "base_selection": Path(words.refinement.__file__),
        "rgb_batch": Path(terminal_rgb_batch.__code__.co_filename), "fd": Path(fd.__file__),
        "goal_contract": Path(goals.__file__), "feasible_goal": Path(feasible_goals.__file__),
        "goal_render_transaction": Path(shared.ReacherGoalImageWrapper._render_target_and_restore.__code__.co_filename)}
    for key, path in modules.items():
        if key != "auditor" and key in alignment["implementation_sha256"] and digest(path) != alignment["implementation_sha256"][key]:
            raise ValueError("audited implementation changed: " + key)
    if (digest(args.policy / "weights.pt") != alignment["weights_sha256"]
            or digest(args.adapter_artifact) != alignment["adapter_sha256"] or source_sha != alignment["source_sha256"]):
        raise ValueError("frozen model, adapter or source changed")
    protocol = {"alignment_reference": str(args.alignment_reference.resolve()), "alignment_sha256": digest(args.alignment_reference),
        "root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": alignment["weights_sha256"], "adapter_sha256": alignment["adapter_sha256"], "source_sha256": source_sha,
        "target_delta": alignment["target_delta"], "goal_contract": args.goal_contract,
        "all17_words_rebuilt": True, "non_greedy_margin": 1e-6,
        "optimizer": native.POLICY, "compiler": compiler.POLICY, "extra_iterations": [8, 32], "damping": .01,
        "cem": {"samples": 128, "steps": 3, "elite": 16, "initial_horizon": 6, "horizon": 2, "seed_offset": 70000,
            "common_random_numbers_across_words": True}, "physical_raw_steps": 30, "prefix_families": list(FAMILIES),
        "physical_branches_per_root": 16, "all_prefix_choices_sealed": True, "all_suffix_choices_sealed_before_terminal_execution": True,
        "physical_labels_used_for_planning": False, "partition_atol": 2e-6,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Source-episode-disjoint numerical-contract confirmation pilot; unequal compute, not P1 or mechanism efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (key + ".py"))
    print("Fresh all8/all17 native prefix-score confirmation; unchanged margin, exact command carry, all5 controls; CUDA only; seal before outcomes.", flush=True)
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
    env = goals.goal_wrapper(args.goal_contract)(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def raw_commands(commands):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * commands)[0].detach().cpu().numpy()
        return p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)

    choices, pending, suffix_choices, terminal_jobs, rows = [], [], [], [], []
    try:
        for root in roots:
            seed = root["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            goal_admission = goals.capture_goal_admission(env, args.goal_contract)
            if any(context[k] != root[k] for k in ("source_episode", "source_start_row")):
                raise RuntimeError("source root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = prepare(context)
            finite = fd._jacobian_and_fd(model, pixels, goal, history, history_len=3, future_horizon=1,
                probe_dims=8, fd_pairs=8, epsilon=.001, action_lower=lower, action_upper=upper, seed=seed, device=device)
            old_bank, original = words.compile_words(model, pixels, goal, history, lower=lower, upper=upper)
            start = samples
            bank, tails, receipt = compiler.compile_words(model, pixels, goal, history, lower=lower, upper=upper,
                seed=seed, progress=lambda i, n: print(json.dumps({"seed": seed, "word_completed": i, "of": n}), flush=True) if i % 4 == 1 or i == n else None)
            gates = {"fd_preflight": finite["median_relative_error"] <= .1,
                "compiler_accounting_exact": samples - start == receipt["model_predict_sample_evaluations"],
                "historical_bank_exact": torch.equal(bank, old_bank), "historical_scores_exact": receipt["original_rows"] == original["proposal_rows"],
                "historical_decision_exact": receipt["historical_decision"] == original["dispatch"],
                "model_partition": receipt["partition_max_abs_error"] <= 2e-6}
            full_cem, _, cem_receipt = cem.search_with_incumbent(lambda params: p0._cost_fn(model, pixels, goal, history, params,
                action_lower=lower, action_upper=upper), future_horizon=6, action_dim=history.shape[-1], device=device,
                samples=128, steps=3, elite=16, seed=seed)
            if not all(gates.values()):
                raise RuntimeError("native prefix preflight failed: " + str(gates))
            choice = {**root, **receipt, "word_params": bank.tolist(), "cem_params": full_cem.tolist(), "cem_receipt": cem_receipt,
                "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                "goal_admission": goal_admission, "fd_receipt": finite, "gates": gates}
            choices.append(choice)
            pending.append((root, context, bank, tails, full_cem, choice))
            print(json.dumps({"seed": seed, "old": receipt["historical_decision"], "native": receipt["native_decision"], "gates": gates}), flush=True)
        write_new(paths["prefix"], {"protocol_sha256": digest(paths["protocol"]), "rows": choices})
        prefix_sha = digest(paths["prefix"])
        for root, context, bank, tails, full_cem, choice in pending:
            seed = root["seed"]
            indices = {"old_dispatch": choice["historical_decision"]["selected_program_index"],
                "native_dispatch": choice["native_decision"]["selected_program_index"],
                "greedy": choice["native_decision"]["same_bank_greedy_index"], "zero": 0}
            branches = [(name, bank[i:i + 1], tails[i]) for name, i in indices.items()]
            branches.append(("cem", full_cem[:, :4], full_cem[:, 4:].tanh()))
            for family, prefix, planned in branches:
                frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                first = raw(prefix)
                advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=frame)
                pixels, goal, history = prepare(advanced)
                emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                start = samples
                cold, cold_receipt = solver.compile_snapshots(model, target, emb, history, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                variants, receipt = native.compile_state_candidates(model, target, emb, history, reference_params=cold,
                    planned_commands=planned, action_lower=lower, action_upper=upper, seed=seed + 70000)
                if samples - start != cold_receipt["model_predict_sample_evaluations"] + receipt["model_predict_sample_evaluations"]:
                    raise RuntimeError("observed suffix accounting drift")
                actions = {"cold": raw(cold), "carry": raw_commands(planned), "native": raw_commands(variants[receipt["selected"]])}
                historical_receipts = {}
                if family == "old_dispatch":
                    i = indices[family]
                    root_pixels, _, root_history = prepare(context)
                    next_emb, next_history = p1._state_after_prefix(model, root_pixels, root_history, bank[i:i + 1], action_lower=lower, action_upper=upper)
                    old_tail, old_receipt = solver.compile_snapshots(model, target, next_emb, next_history, future_horizon=2,
                        action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
                    if old_receipt != choice["original_rows"][i]["suffix_receipt"]:
                        raise RuntimeError("historical imagined suffix reconstruction drift")
                    local, local_receipt = continuation.compile_continuation(model, target, emb, history, planned_suffix=old_tail,
                        action_lower=lower, action_upper=upper)
                    old_variants, historical = native.compile_state_candidates(model, target, emb, history,
                        reference_params=local[local_receipt["selected"]], action_lower=lower, action_upper=upper, seed=seed + 70000)
                    actions["historical"] = raw_commands(old_variants[historical["selected"]])
                    historical_receipts = {"historical_local_receipt": local_receipt, "historical_receipt": historical}
                record = {**root, "family": family, "prefix_index": indices.get(family), "prefix_actions_raw": first.tolist(),
                    "prefix_boundaries": boundaries, "execution_identity": identity,
                    "prefix_rgb_cost": float((emb[0, -1] - target).square().mean()), "planned_commands": planned.tolist(),
                    "native_commands": variants[receipt["selected"]].tolist(), "receipt": receipt, "cold_receipt": cold_receipt,
                    "suffix_actions_raw": {name: value.tolist() for name, value in actions.items()}, **historical_receipts}
                suffix_choices.append(record)
                terminal_jobs.append((root, advanced, actions, record))
                print(json.dumps({"seed": seed, "family": family, "selected": receipt["selected"]}), flush=True)
        write_new(paths["suffix"], {"protocol_sha256": digest(paths["protocol"]), "prefix_selection_sha256": prefix_sha, "rows": suffix_choices})
        suffix_sha = digest(paths["suffix"])
        print("All40 observed suffix decisions sealed; begin128 physical terminal branches.", flush=True)
        for root in roots:
            methods, branches = {}, {}
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
                    rgb = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                    complete = np.concatenate((np.asarray(record["prefix_actions_raw"], np.float32), suffix))
                    if complete.shape != (30, 2) or not np.isfinite(complete).all() or np.abs(complete).max() > 1 + 1e-6 or not np.isfinite([distance, rgb]).all():
                        raise RuntimeError("bounded complete physical witness/metric contract failed")
                    methods[family + "_" + name] = {"action_witness_raw": complete.tolist(), "physical": {"final_qpos_distance": distance, "final_rgb_cost": rgb}}
                branches[family] = record
            gates = {"all16_physical_branches": len(methods) == 16 and set(branches) == set(FAMILIES)}
            row = {**root, "methods": methods, "branches": branches, "gates": gates}
            write_new(paths["roots"] / f"root_{root['seed']}.json", row)
            rows.append(row)
            print(json.dumps({"seed": root["seed"], "distances": {k: v["physical"]["final_qpos_distance"] for k, v in methods.items() if k.endswith(("_native", "_historical"))}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    metrics = {name: [r["methods"][name]["physical"]["final_qpos_distance"] for r in rows] for name in rows[0]["methods"]}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model), "all_declared_roots": [r["seed"] for r in rows] == seeds,
        "explicit_goal_contract": args.goal_contract in goals.GOAL_CONTRACTS,
        "all_goal_admissions": all(c["goal_admission"] is not None for c in choices) if args.goal_contract == "joint-feasible-v1" else all(c["goal_admission"] is None for c in choices),
        "source_episodes_disjoint": not ({r["source_episode"] for r in rows} & excluded), "all17_words_every_root": all(len(c["word_records"]) == 17 for c in choices),
        "all_core_equivalence_gates": all(all(c["gates"].values()) for c in choices), "all_physical_gates": all(all(r["gates"].values()) for r in rows),
        "prefix_seal_unchanged": digest(paths["prefix"]) == prefix_sha, "suffix_seal_unchanged": digest(paths["suffix"]) == suffix_sha,
        "alignment_unchanged": digest(args.alignment_reference) == protocol["alignment_sha256"],
        "exclusions_unchanged": all(digest(Path(b["path"])) == b["sha256"] for b in bindings),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    aggregate = {"roots": 8, "imagined_words": 136, "physical_branches": 128, "mean_distance": {k: float(np.mean(v)) for k, v in metrics.items()},
        "historical_witnesses": sum(c["historical_decision"]["reconfiguration_witness_found"] for c in choices),
        "native_witnesses": sum(c["native_decision"]["reconfiguration_witness_found"] for c in choices),
        "prefix_choices_changed": sum(c["historical_decision"]["selected_program_index"] != c["native_decision"]["selected_program_index"] for c in choices),
        "comparisons": {name: paired_difference(metrics["native_dispatch_native"], metrics[name]) for name in
            ("old_dispatch_historical", "old_dispatch_native", "greedy_native", "zero_native", "cem_native")}}
    write_new(output, {"schema": "reacher_d2p_state_native_confirmation.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": prefix_sha, "suffix_selection_sha256": suffix_sha,
        "gates": gates, "aggregate": aggregate, "rows": rows, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
