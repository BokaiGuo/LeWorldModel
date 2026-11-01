"""Known-root endpoint-only selector ablation; never an alternative P1 gate.

The current P1 selector intentionally prioritizes a non-greedy witness. This
audit selects the global model endpoint minimum instead, without inspecting
physical outcomes. All8 roots and all32 physical branches remain in evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_state_suffix as native
from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation import load_alignment
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference


def select_global_word(rows, *, non_greedy_margin=1e-6):
    if (len(rows) != 17 or [r["proposal_index"] for r in rows] != list(range(17))
            or not math.isfinite(non_greedy_margin) or non_greedy_margin < 0
            or any(not math.isfinite(float(r[k])) or float(r[k]) < 0 for r in rows for k in ("immediate_goal_cost", "refined_goal_cost"))):
        raise ValueError("all17 ordered finite nonnegative model scores required")
    selected = min(range(17), key=lambda i: (rows[i]["refined_goal_cost"], i))
    return {"selected_program_index": selected, "policy": "global_model_endpoint_minimum_all17_v1",
        "model_cost": rows[selected]["refined_goal_cost"], "physical_labels_used": False,
        "immediate_model_non_greedy": selected != 0 and rows[selected]["immediate_goal_cost"]
            >= rows[0]["immediate_goal_cost"] + non_greedy_margin,
        "scope": "Endpoint-only selector ablation, not a P1 admission or reconfiguration-mechanism verdict."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref = load_alignment(args.reference)
    report = json.loads(args.reference.read_text())
    prefixes = json.loads(args.reference.with_suffix(".selection.json").read_text())
    old_path = Path(ref["reference_path"])
    old = json.loads(old_path.read_text())
    old_prefix_path = old_path.with_suffix(".selection.json")
    if digest(old_path) != ref["reference_sha256"] or digest(old_prefix_path) != old["selection_sha256"]:
        raise ValueError("original word action witness seal changed")
    banks = {r["seed"]: r for r in json.loads(old_prefix_path.read_text())["rows"]}
    source = {r["seed"]: r for r in report["rows"]}
    choices = sorted(prefixes["rows"], key=lambda r: r["seed"])
    if sorted(banks) != sorted(source) or [c["seed"] for c in choices] != sorted(source):
        raise ValueError("complete unique root identities required")
    output = args.output.resolve()
    paths = {k: output.with_suffix(v) for k, v in (("protocol", ".protocol.json"), ("prefix", ".selection.json"),
        ("suffix", ".suffix.selection.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("frozen model or action statistics changed")
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    modules = {"auditor": Path(__file__), "words": Path(words.__file__), "state_suffix": Path(native.__file__),
        "box": Path(native.box.__file__), "qp": Path(native.box.solve_box_quadratic.__code__.co_filename),
        "p0": Path(p0.__file__), "prefix": Path(p1.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "cem": Path(native.cem.__file__), "source_support": Path(support.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "rgb_batch": Path(terminal_rgb_batch.__code__.co_filename)}
    if any(digest(path) != ref["implementation_sha256"][k] for k, path in modules.items() if k != "auditor"):
        raise ValueError("audited numerical implementation changed")
    protocol = {"root_seeds": sorted(source), "reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "word_prefix_sha256": digest(old_prefix_path), "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"],
        "source_sha256": ref["source_sha256"], "target_delta": ref["target_delta"], "non_greedy_margin": 1e-6,
        "physical_raw_steps": 30, "physical_branches": 32, "physical_labels_used_for_planning": False,
        "all_suffix_choices_sealed_before_terminal_execution": True,
        "implementation_sha256": {k: digest(path) for k, path in modules.items()},
        "scope": "Known-root endpoint-only selector ablation, unequal compute; does not change P1 or theory verdict."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (k + ".py"))
    selected = [{"seed": c["seed"], "source_episode": c["source_episode"], "source_start_row": c["source_start_row"],
        "decision": select_global_word(c["score_rows"]), "planned_commands": c["word_records"][select_global_word(c["score_rows"])["selected_program_index"]]["planned_commands"]}
        for c in choices]
    write_new(paths["prefix"], {"protocol_sha256": digest(paths["protocol"]), "rows": selected})
    prefix_sha = digest(paths["prefix"])
    print("All8 endpoint-only global decisions sealed from original136 model scores; no physical labels used. CUDA only; keep native selector as exact replay baseline.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    dataset, sha = shared.receipt_source(args.adapter_artifact)
    if sha != ref["source_sha256"] or support._sha256(dataset) != sha:
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

    def raw_commands(commands):
        normalized = ((lower + upper) / 2 + (upper - lower) / 2 * commands)[0].detach().cpu().numpy()
        return p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)

    jobs, suffixes, rows = [], [], []
    try:
        for choice in selected:
            seed = choice["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[k] != choice[k] for k in ("source_episode", "source_start_row")):
                raise RuntimeError("source identity changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            identity = p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
            bank = torch.tensor(banks[seed]["word_params"], device=device, dtype=torch.float32)
            i = choice["decision"]["selected_program_index"]
            first = p0._to_raw_actions(bank[i:i + 1], mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]
            advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=frame)
            pixels, goal, history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"],
                np.asarray(advanced["history_action_blocks"], np.float32), device)
            emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            planned = torch.tensor(choice["planned_commands"], device=device, dtype=history.dtype)
            start = samples
            cold, cold_receipt = solver.compile_snapshots(model, target, emb, history, future_horizon=2,
                action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
            variants, receipt = native.compile_state_candidates(model, target, emb, history, reference_params=cold,
                planned_commands=planned, action_lower=lower, action_upper=upper, seed=seed + 70000)
            if samples - start != cold_receipt["model_predict_sample_evaluations"] + receipt["model_predict_sample_evaluations"]:
                raise RuntimeError("observed numerical accounting drift")
            raw_cold = p0._to_raw_actions(cold, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]
            actions = {"cold": raw_cold, "carry": raw_commands(planned), "native": raw_commands(variants[receipt["selected"]])}
            record = {**choice, "prefix_actions_raw": first.tolist(), "prefix_boundaries": boundaries, "execution_identity": identity,
                "prefix_rgb_cost": float((emb[0, -1] - target).square().mean()), "receipt": receipt,
                "suffix_actions_raw": {k: v.tolist() for k, v in actions.items()}}
            suffixes.append(record)
            jobs.append((choice, context, advanced, actions, record))
            print(json.dumps({"seed": seed, "global_word": i, "selected": receipt["selected"]}), flush=True)
        write_new(paths["suffix"], {"protocol_sha256": digest(paths["protocol"]), "prefix_selection_sha256": prefix_sha, "rows": suffixes})
        suffix_sha = digest(paths["suffix"])
        for choice, context, advanced, actions, record in jobs:
            seed, methods = choice["seed"], {}
            for name, tail in actions.items():
                execution.restore_execution_snapshot(env, seed=seed, snapshot=advanced["execution_snapshot"], root_identity="sealed_observation_exact_state")
                execution._execute_raw_steps(env, tail)
                physics = env.unwrapped.env.physics
                qpos = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                prepared = processor._prepare_info(terminal_rgb_batch(env.render(), advanced["goal"], mean.size))
                rgb = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                complete = np.concatenate((np.asarray(record["prefix_actions_raw"], np.float32), tail))
                if complete.shape != (30, 2) or not np.isfinite(complete).all() or np.abs(complete).max() > 1 + 1e-6 or not np.isfinite([qpos, rgb]).all():
                    raise RuntimeError("bounded complete physical witness required")
                methods[name] = {"physical": {"final_qpos_distance": qpos, "final_rgb_cost": rgb}, "action_witness_raw": complete.tolist()}
            baseline = source[seed]["methods"]["native_dispatch_native"]
            execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            execution._execute_raw_steps(env, np.asarray(baseline["action_witness_raw"], np.float32))
            physics = env.unwrapped.env.physics
            baseline_distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
            prepared = processor._prepare_info(terminal_rgb_batch(env.render(), context["goal"], mean.size))
            baseline_rgb = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
            same = choice["decision"]["selected_program_index"] == next(c for c in choices if c["seed"] == seed)["native_decision"]["selected_program_index"]
            gates = {"baseline_distance_exact": baseline_distance == baseline["physical"]["final_qpos_distance"],
                "baseline_rgb_exact": baseline_rgb == baseline["physical"]["final_rgb_cost"],
                "unchanged_choice_exact": not same or methods["native"] == baseline}
            if not all(gates.values()):
                raise RuntimeError("baseline/same-selector exact replay failed: " + str(gates))
            row = {**choice, "methods": methods, "native_selector_baseline": baseline, "branch": record, "gates": gates}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "distances": {k: v["physical"]["final_qpos_distance"] for k, v in methods.items()}, "baseline": baseline_distance, "gates": gates}), flush=True)
    finally:
        model.predict = predict
        env.close()
    metric = {k: [r["methods"][k]["physical"]["final_qpos_distance"] for r in rows] for k in ("cold", "carry", "native")}
    baseline = [r["native_selector_baseline"]["physical"]["final_qpos_distance"] for r in rows]
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == sorted(source), "weights_unchanged": before == shared._parameter_hash(model),
        "all_exact_replays": all(all(r["gates"].values()) for r in rows), "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "prefix_seal_unchanged": digest(paths["prefix"]) == prefix_sha, "suffix_seal_unchanged": digest(paths["suffix"]) == suffix_sha,
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][k] for k, path in modules.items())}
    aggregate = {"mean_distance": {k: float(np.mean(v)) for k, v in metric.items()}, "baseline_mean_distance": float(np.mean(baseline)),
        "global_vs_native_selector": paired_difference(metric["native"], baseline), "physical_branches": 32}
    write_new(output, {"schema": "reacher_d2p_native_global_word_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": prefix_sha, "suffix_selection_sha256": suffix_sha,
        "rows": rows, "gates": gates, "aggregate": aggregate, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
