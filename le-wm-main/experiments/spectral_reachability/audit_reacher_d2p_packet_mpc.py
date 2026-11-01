"""Isolate feedback cadence and carried-plan retention on all eight known roots.

The frozen RGB planner sees a new observation after EACH five-step packet.
Compare cold CEM replanning, the same search with an exact carried-plan
incumbent, and the original complete six-packet open-loop CEM. This changes
feedback and compute, not P1 eligibility, model weights, or physical horizon.
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
from experiments.spectral_reachability import reacher_d2p_packet_support as support
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation import load_alignment
from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import paired_difference


def choose_plan(cost_fn, searched, carried=None):
    """Singleton-rescore both exact unconstrained plans; ties retain carry.

    No inverse tanh, no physics labels, and no injection into CEM generations.
    A carried tail is rescored at the CURRENT RGB context, never its old score.
    """
    if searched.ndim != 3 or searched.shape[0] != 1 or min(searched.shape[1:]) < 1:
        raise ValueError("one nonempty H-by-action plan required")
    plans = {"searched": searched}
    if carried is not None:
        if carried.shape != searched.shape or carried.device != searched.device or carried.dtype != searched.dtype:
            raise ValueError("carried plan shape/device/dtype must match current search")
        plans["carried"] = carried
    costs = {}
    with torch.no_grad():
        for name, plan in plans.items():
            if not torch.isfinite(plan).all():
                raise ValueError("finite plans required")
            value = cost_fn(plan)
            if value.shape != (1,) or not torch.isfinite(value).all() or bool((value < 0).any()):
                raise ValueError("one finite nonnegative current-context cost required")
            costs[name] = float(value[0])
    selected = min(costs, key=lambda k: (costs[k], k != "carried"))
    return plans[selected].detach().clone(), {"selected": selected, "current_model_costs": costs,
        "physical_labels_used": False, "carry_rescored_on_current_rgb": carried is not None,
        "inverse_tanh_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref = load_alignment(args.reference)
    report = json.loads(args.reference.read_text())
    sources = sorted(report["rows"], key=lambda r: r["seed"])
    original_path = Path(ref["reference_path"])
    original = json.loads(original_path.read_text())
    original_choices_path = original_path.with_suffix(".selection.json")
    if digest(original_path) != ref["reference_sha256"] or digest(original_choices_path) != original["selection_sha256"]:
        raise ValueError("original CEM decision seals changed")
    original_choices = {r["seed"]: r for r in json.loads(original_choices_path.read_text())["rows"]}
    p0 = words.p0
    modules = {"auditor": Path(__file__), "p0": Path(p0.__file__), "words": Path(words.__file__),
        "cem": Path(cem.__file__), "context": Path(execution.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename),
        "rgb_batch": Path(terminal_rgb_batch.__code__.co_filename)}
    if any(digest(p) != ref["implementation_sha256"][k] for k, p in modules.items() if k != "auditor"):
        raise ValueError("audited neural/execution implementation changed")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("frozen checkpoint/normalization changed")
    output = args.output.resolve()
    paths = {k: output.with_suffix(v) for k, v in (("protocol", ".protocol.json"), ("stages", ".stages"),
        ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "root_seeds": [r["seed"] for r in sources], "weights_sha256": ref["weights_sha256"],
        "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"], "target_delta": ref["target_delta"],
        "initial_horizon": 6, "packet_raw_steps": 5, "physical_raw_steps": 30,
        "modes": ["cold", "carry"], "cem": {"samples": 128, "steps": 3, "elite": 16},
        "search_seed": "root_seed + 90000 * stage", "all_stage_decisions_sealed_before_stage_execution": True,
        "all_final_packet_decisions_sealed_before_any_terminal_evaluation": True,
        "physical_labels_used_for_planning": False, "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "Known-root feedback-cadence and exact-carry diagnostic, unequal compute; no P1 or theory verdict."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    for key in ("stages", "roots", "implementation"):
        paths[key].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (key + ".py"))
    print("All8 packet feedback audit: observe RGB every5 raw steps; shrinking6..1 CEM; exact carry vs cold; original open-loop exact replay; CUDA only; stage seals before execution.", flush=True)
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

    def tensors(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def physical(context):
        physics = env.unwrapped.env.physics
        distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
        prepared = processor._prepare_info(terminal_rgb_batch(context["root_frame"], context["goal"], mean.size))
        with torch.no_grad():
            rgb = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
        if not np.isfinite([distance, rgb]).all() or min(distance, rgb) < 0:
            raise RuntimeError("finite nonnegative terminal metrics required")
        return {"final_qpos_distance": distance, "final_rgb_cost": rgb}

    sessions, roots, initial_plans, stage_files, rows = {}, {}, {}, [], []
    try:
        for source in sources:
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[k] != source[k] for k in ("source_episode", "source_start_row")):
                raise RuntimeError("source identity changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            roots[seed] = context
            for mode in protocol["modes"]:
                sessions[seed, mode] = {"context": context, "carried": None, "actions": [], "stages": [], "boundaries": []}
        parent_sha = digest(paths["protocol"])
        for stage in range(6):
            pending, decisions = [], []
            for source in sources:
                seed = source["seed"]
                for mode in protocol["modes"]:
                    state = sessions[seed, mode]
                    pixels, goal, history = tensors(state["context"])
                    cost_fn = lambda p: p0._cost_fn(model, pixels, goal, history, p, action_lower=lower, action_upper=upper)
                    start = samples
                    searched, _, receipt = cem.search_with_incumbent(cost_fn, future_horizon=6 - stage, action_dim=history.shape[-1],
                        device=device, samples=128, steps=3, elite=16, seed=seed + 90000 * stage)
                    if stage == 0:
                        old = original_choices[seed]
                        if searched.tolist() != old["cem_params"] or receipt != old["cem_receipt"]:
                            raise RuntimeError("original full CEM trajectory/actions changed")
                        initial_plans[seed] = searched.detach().clone()
                    plan, selection = choose_plan(cost_fn, searched, state["carried"] if mode == "carry" else None)
                    actual = samples - start
                    expected = (384 + len(selection["current_model_costs"])) * (6 - stage)
                    if actual != expected:
                        raise RuntimeError("per-stage neural compute accounting drift")
                    actions = raw(plan[:, :1])
                    if actions.shape != (5, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
                        raise RuntimeError("finite bounded five-step action required")
                    decision = {"seed": seed, "mode": mode, "stage": stage, "remaining_horizon": 6 - stage,
                        "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                        "root_rgb_sha256": state["context"]["root_rgb_sha256"],
                        "history_rgb_sha256": state["context"]["history_rgb_sha256"],
                        "searched_params": searched.tolist(), "carried_params": state["carried"].tolist() if mode == "carry" and state["carried"] is not None else None,
                        "chosen_params": plan.tolist(), "first_packet_raw": actions.tolist(), "selection": selection,
                        "search_receipt": receipt, "actual_model_predict_samples": actual}
                    decisions.append(decision)
                    pending.append((source, mode, state, plan, actions, decision))
            path = paths["stages"] / f"stage_{stage}.json"
            write_new(path, {"protocol_sha256": digest(paths["protocol"]), "previous_stage_sha256": parent_sha, "rows": decisions})
            parent_sha = digest(path)
            stage_files.append({"stage": stage, "path": str(path), "sha256": parent_sha})
            print(json.dumps({"stage": stage, "all16_decisions_sealed": True}), flush=True)
            for source, mode, state, plan, actions, decision in pending:
                seed, context = source["seed"], state["context"]
                frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                p0._verify_execution_identity(frame, goal, context["root_frame"], context["goal"])
                advanced, boundaries = words.advance_program(env, context, actions, verified_root_frame=frame)
                state["context"] = advanced
                state["carried"] = plan[:, 1:].detach().clone() if stage < 5 else None
                state["actions"].extend(actions.tolist())
                state["stages"].append(decision)
                state["boundaries"].extend(boundaries)
                if stage == 5:
                    state["terminal"] = physical(advanced)
        # No terminal labels were used above to select or change a later plan.
        for source in sources:
            seed, context = source["seed"], roots[source["seed"]]
            methods, gates = {}, {}
            for mode in protocol["modes"]:
                state = sessions[seed, mode]
                actions = np.asarray(state["actions"], np.float32)
                frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                advanced, boundaries = words.advance_program(env, context, actions, verified_root_frame=frame)
                replay = physical(advanced)
                gates[mode + "_full_physical_replay_exact"] = replay == state["terminal"]
                gates[mode + "_all_boundary_rgb_replay_exact"] = [b["root_rgb_sha256"] for b in boundaries] == [b["root_rgb_sha256"] for b in state["boundaries"]]
                gates[mode + "_initial_packet_exact"] = actions[:5].tolist() == raw(initial_plans[seed])[:5].tolist()
                methods[mode] = {"action_witness_raw": actions.tolist(), "physical": replay,
                    "stages": state["stages"], "boundaries": state["boundaries"]}
            frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
            actions = raw(initial_plans[seed])
            advanced, boundaries = words.advance_program(env, context, actions, verified_root_frame=frame)
            baseline = {"action_witness_raw": actions.tolist(), "physical": physical(advanced)}
            gates["original_openloop_exact"] = baseline == source["methods"]["cem_carry"]
            methods["openloop"] = baseline
            if not all(gates.values()):
                raise RuntimeError("feedback/replay integrity failed: " + str(gates))
            row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                "methods": methods, "gates": gates}
            write_new(paths["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "distances": {k: m["physical"]["final_qpos_distance"] for k, m in methods.items()}, "gates": gates}), flush=True)
    finally:
        model.predict = predict
        env.close()
    metrics = {k: [r["methods"][k]["physical"]["final_qpos_distance"] for r in rows] for k in ("cold", "carry", "openloop")}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all8_roots": [r["seed"] for r in rows] == protocol["root_seeds"], "all_exact_replays": all(all(r["gates"].values()) for r in rows),
        "all_stage_seals_unchanged": all(digest(Path(s["path"])) == s["sha256"] for s in stage_files),
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    aggregate = {"mean_distance": {k: float(np.mean(v)) for k, v in metrics.items()}, "physical_branches": 24,
        "carry_vs_openloop": paired_difference(metrics["carry"], metrics["openloop"]),
        "cold_vs_openloop": paired_difference(metrics["cold"], metrics["openloop"]),
        "carry_vs_cold": paired_difference(metrics["carry"], metrics["cold"])}
    write_new(output, {"schema": "reacher_d2p_packet_mpc_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "stage_files": stage_files, "rows": rows, "gates": gates,
        "aggregate": aggregate, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
