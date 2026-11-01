"""All-root CEM coverage audit, with fixed microbatch and exact old replay.

Increase search to1024x12 with64 elites, not P1 thresholds or physical time.
Compare full6-token search and the same native prefix's2-token suffix. Extra
compute is explicit; this is not a reconfiguration or efficiency result.
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
from experiments.spectral_reachability import reacher_d2p_state_suffix as native
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation import load_alignment
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference


def microbatched_cost(cost_fn, params, *, batch_size=128):
    if params.ndim != 3 or params.shape[0] < 1 or batch_size < 1:
        raise ValueError("nonempty candidate bank and positive microbatch required")
    values = []
    for start in range(0, params.shape[0], batch_size):
        part = params[start:start + batch_size]
        value = cost_fn(part)
        if value.shape != (len(part),) or not torch.isfinite(value).all() or bool((value < 0).any()):
            raise ValueError("finite nonnegative cost per candidate required")
        values.append(value)
    return torch.cat(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref = load_alignment(args.reference)
    report = json.loads(args.reference.read_text())
    base_path = Path(ref["reference_path"])
    base = json.loads(base_path.read_text())
    base_prefix = base_path.with_suffix(".selection.json")
    if digest(base_path) != ref["reference_sha256"] or digest(base_prefix) != base["selection_sha256"]:
        raise ValueError("original CEM/action seals changed")
    roots = sorted(report["rows"], key=lambda r: r["seed"])
    old_choices = {r["seed"]: r for r in json.loads(base_prefix.read_text())["rows"]}
    output = args.output.resolve()
    paths = {k: output.with_suffix(v) for k, v in (("protocol", ".protocol.json"), ("selection", ".selection.json"),
        ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("frozen model/adapter changed")
    p0, p1, cem = words.p0, words.p1, native.cem
    modules = {"auditor": Path(__file__), "p0": Path(p0.__file__), "prefix": Path(p1.__file__), "context": Path(execution.__file__),
        "words": Path(words.__file__), "state_suffix": Path(native.__file__), "cem": Path(cem.__file__), "box": Path(native.box.__file__),
        "source_support": Path(support.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "rgb_batch": Path(terminal_rgb_batch.__code__.co_filename)}
    if any(digest(p) != ref["implementation_sha256"][k] for k, p in modules.items() if k != "auditor"):
        raise ValueError("historical neural/execution implementation changed")
    protocol = {"reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference), "root_seeds": [r["seed"] for r in roots],
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "target_delta": ref["target_delta"], "physical_raw_steps": 30, "physical_branches": 32, "microbatch": 128,
        "original_cem": {"samples": 128, "steps": 3, "elite": 16}, "wide_cem": {"samples": 1024, "steps": 12, "elite": 64},
        "suffix_seed_offset": 70000, "physical_labels_used_for_planning": False, "all_choices_sealed_before_terminal_execution": True,
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "Known-root search coverage diagnostic; substantially unequal compute, no P1 or theory verdict."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k, p in modules.items():
        shutil.copyfile(p, paths["implementation"] / (k + ".py"))
    print("All8 CEM coverage audit: fixed128 microbatch;1024x12/64 elites vs original128x3/16; full6 and fixed-prefix suffix2; CUDA only. Seal all choices before terminal outcomes.", flush=True)
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

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    choices, pending, rows = [], [], []
    try:
        for old in roots:
            seed = old["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[k] != old[k] for k in ("source_episode", "source_start_row")):
                raise RuntimeError("source root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = prepare(context)
            cost_fn = lambda params: p0._cost_fn(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)
            old_full, _, original = cem.search_with_incumbent(cost_fn, future_horizon=6, action_dim=history.shape[-1], device=device,
                samples=128, steps=3, elite=16, seed=seed)
            if old_full.tolist() != old_choices[seed]["cem_params"] or original != old_choices[seed]["cem_receipt"]:
                raise RuntimeError("original complete CEM trajectory failed exact replay")
            wide_full, _, full_receipt = cem.search_with_incumbent(lambda p: microbatched_cost(cost_fn, p), future_horizon=6,
                action_dim=history.shape[-1], device=device, samples=1024, steps=12, elite=64, seed=seed)
            with torch.no_grad():
                full_costs = {name: float(cost_fn(value)[0]) for name, value in (("original", old_full), ("wide", wide_full))}
            frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
            original_native = np.asarray(old["methods"]["native_dispatch_native"]["action_witness_raw"], np.float32)
            advanced, boundaries = words.advance_program(env, context, original_native[:20], verified_root_frame=frame)
            pixels, goal, history = prepare(advanced)
            emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            suffix_cost_fn = lambda p: native.state_costs(model, target, emb, history, p, action_lower=lower, action_upper=upper)
            wide_suffix, _, suffix_receipt = cem.search_with_incumbent(lambda p: microbatched_cost(suffix_cost_fn, p), future_horizon=2,
                action_dim=history.shape[-1], device=device, samples=1024, steps=12, elite=64, seed=seed + 70000)
            branch = old["branches"]["native_dispatch"]
            native_commands = torch.tensor(branch["native_commands"], device=device, dtype=history.dtype)
            with torch.no_grad():
                native_terminal = native.box.terminal_commands(model, emb, history, native_commands, action_lower=lower, action_upper=upper)[0]
                native_cost = float((p0._planning_coordinate(model, native_terminal, role="predicted") - p0._planning_coordinate(model, target, role="goal")).square().mean())
                suffix_costs = {"native": native_cost, "wide": float(suffix_cost_fn(wide_suffix)[0])}
            if (abs(native_cost - branch["receipt"]["model_costs"][branch["receipt"]["selected"]]) > 2e-6
                    or boundaries != branch["prefix_boundaries"]):
                raise RuntimeError("observed prefix/numerical incumbent changed")
            actions = {"full_original": raw(old_full), "full_wide": raw(wide_full), "fixed_native": original_native,
                "fixed_wide": np.concatenate((original_native[:20], raw(wide_suffix)))}
            choice = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "full_costs": full_costs, "suffix_costs": suffix_costs,
                "full_selected": min(full_costs, key=lambda k: (full_costs[k], k != "original")),
                "suffix_selected": min(suffix_costs, key=lambda k: (suffix_costs[k], k != "native")),
                "full_receipt": full_receipt, "suffix_receipt": suffix_receipt, "original_receipt": original,
                "action_witnesses_raw": {k: v.tolist() for k, v in actions.items()}, "physical_labels_used": False}
            write_new(paths["roots"] / f"selection_{seed}.json", choice)
            choices.append(choice)
            pending.append((old, context, actions, choice))
            print(json.dumps({"seed": seed, "full_costs": full_costs, "suffix_costs": suffix_costs}), flush=True)
        write_new(paths["selection"], {"protocol_sha256": digest(paths["protocol"]), "rows": choices})
        selection_sha = digest(paths["selection"])
        for old, context, actions, choice in pending:
            seed, methods = old["seed"], {}
            for name, action in actions.items():
                execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                execution._execute_raw_steps(env, action)
                physics = env.unwrapped.env.physics
                distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                prepared = processor._prepare_info(terminal_rgb_batch(env.render(), context["goal"], mean.size))
                rgb = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                if action.shape != (30, 2) or not np.isfinite(action).all() or np.abs(action).max() > 1 + 1e-6 or not np.isfinite([distance, rgb]).all():
                    raise RuntimeError("bounded complete physical witness required")
                methods[name] = {"action_witness_raw": action.tolist(), "physical": {"final_qpos_distance": distance, "final_rgb_cost": rgb}}
            gates = {"native_complete_replay_exact": methods["fixed_native"] == old["methods"]["native_dispatch_native"]}
            if not all(gates.values()):
                raise RuntimeError("native physical baseline changed")
            row = {"seed": seed, "choice": choice, "methods": methods, "gates": gates}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "distances": {k: m["physical"]["final_qpos_distance"] for k, m in methods.items()}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    metric = {k: [r["methods"][k]["physical"]["final_qpos_distance"] for r in rows] for k in rows[0]["methods"]}
    for family in ("full", "suffix"):
        metric[family + "_model_best"] = [r["methods"][("full_" + r["choice"]["full_selected"]) if family == "full"
            else ("fixed_" + r["choice"]["suffix_selected"])]["physical"]["final_qpos_distance"] for r in rows]
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == ref["root_seeds"], "weights_unchanged": before == shared._parameter_hash(model),
        "all_exact_replays": all(all(r["gates"].values()) for r in rows), "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "selection_seal_unchanged": digest(paths["selection"]) == selection_sha,
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    aggregate = {"mean_distance": {k: float(np.mean(v)) for k, v in metric.items()}, "physical_branches": 32,
        "wide_full_vs_original": paired_difference(metric["full_wide"], metric["full_original"]),
        "wide_suffix_vs_native": paired_difference(metric["fixed_wide"], metric["fixed_native"])}
    write_new(output, {"schema": "reacher_d2p_cem_budget_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": selection_sha, "rows": rows, "gates": gates,
        "aggregate": aggregate, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
