"""Paired host-CUDA task-construction intervention, retaining all16 roots.

Legacy and joint-feasible goals use identical frozen-model CEM settings.
Reconstruct every historical search and open-loop physical outcome exactly.
All32 complete plans are sealed before any terminal evaluation. A changed
task is never treated as repairing old P1 ranking or as equal-task efficacy.
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
from experiments.spectral_reachability import reacher_d2p_feasible_goal as goals
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart import terminal_rgb_batch
from experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector import summarize as validate_reference


def load_cem_choices(report, prefix):
    """Known-root audits inherit CEM seals; fresh confirmations own them."""
    rows = prefix["rows"]
    required = {"cem_params", "cem_receipt", "root_rgb_sha256", "goal_rgb_sha256"}
    if not all(required.issubset(row) for row in rows):
        if report.get("schema") != "reacher_d2p_state_native_selector_audit.v1":
            raise ValueError("fresh CEM choices must be explicit")
        protocol = report["sealed_protocol"]
        path = Path(protocol["reference_path"])
        parent = json.loads(path.read_text())
        seal = path.with_suffix(".selection.json")
        if (digest(path) != protocol["reference_sha256"] or digest(seal) != parent["selection_sha256"]
                or parent.get("status") != "completed" or not parent.get("gates")
                or not all(v is True for v in parent["gates"].values())):
            raise ValueError("inherited CEM report/decision seal changed")
        rows = json.loads(seal.read_text())["rows"]
    choices = {row["seed"]: row for row in rows}
    if (len(choices) != len(rows) or sorted(choices) != sorted(report["sealed_protocol"]["root_seeds"])
            or not all(required.issubset(row) for row in rows)):
        raise ValueError("all unique explicit CEM/root witnesses required")
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reference) != 2:
        raise ValueError("both complete known and independently confirmed all8 references required")
    sources, refs = [], []
    for path in args.reference:
        report = json.loads(path.read_text())
        prefix_path, suffix_path, protocol_path = (path.with_suffix(s) for s in (".selection.json", ".suffix.selection.json", ".protocol.json"))
        prefix, suffix = json.loads(prefix_path.read_text()), json.loads(suffix_path.read_text())
        if (digest(prefix_path) != report["selection_sha256"] or digest(suffix_path) != report["suffix_selection_sha256"]
                or prefix["protocol_sha256"] != digest(protocol_path) or suffix["protocol_sha256"] != digest(protocol_path)
                or suffix["prefix_selection_sha256"] != digest(prefix_path)
                or json.loads(protocol_path.read_text()) != report["sealed_protocol"]):
            raise ValueError("reference protocol/decision seals changed")
        validate_reference(report, prefix, suffix)
        choices = load_cem_choices(report, prefix)
        sources.extend((row, choices[row["seed"]]) for row in report["rows"])
        refs.append({"path": str(path.resolve()), "sha256": digest(path), "protocol": report["sealed_protocol"]})
    sources.sort(key=lambda pair: pair[0]["seed"])
    if len(sources) != 16 or len({r["seed"] for r, _ in sources}) != 16:
        raise ValueError("all16 unique reference roots required")
    ref = refs[0]["protocol"]
    for record in refs:
        if any(record["protocol"][k] != ref[k] for k in ("weights_sha256", "adapter_sha256", "source_sha256", "target_delta")):
            raise ValueError("common frozen model/source/goal request required")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("frozen model/normalizer changed")
    p0 = words.p0
    modules = {"auditor": Path(__file__), "goal_repair": Path(goals.__file__), "legacy_goal": Path(shared.ReacherGoalImageWrapper.__module__.replace(".", "/") + ".py"),
        "p0": Path(p0.__file__), "context": Path(execution.__file__), "cem": Path(cem.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for name, path in modules.items():
        if name != "auditor" and name in ref["implementation_sha256"] and digest(path) != ref["implementation_sha256"][name]:
            raise ValueError("historical implementation changed: " + name)
    output = args.output.resolve()
    paths = {k: output.with_suffix(v) for k, v in (("protocol", ".protocol.json"), ("selection", ".selection.json"),
        ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in [output, *paths.values()]):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"references": refs, "root_seeds": [r["seed"] for r, _ in sources], "target_delta": ref["target_delta"],
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "cem": {"horizon": 6, "samples": 128, "steps": 3, "elite": 16, "seed": "root_seed"},
        "all32_plans_sealed_before_terminal_evaluation": True, "physical_labels_used_for_planning": False,
        "task_change": "bounded hinge full-offset sign reflection, continuous shoulder periodic wrap unchanged",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "All16 known-root TASK-construction repair diagnostic, not equal-task improvement, not P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for name, path in modules.items():
        shutil.copyfile(path, paths["implementation"] / (name + ".py"))
    print("All16 original roots retained: legacy vs actual-joint-limit feasible goals; same frozen CEM128x3; host CUDA only; all32 plans seal before outcomes. Changed tasks do NOT repair historical P1.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    from experiments.spectral_reachability.reacher_d2p_packet_support import _sha256
    if source_sha != ref["source_sha256"] or _sha256(dataset) != source_sha:
        raise ValueError("complete source SHA changed")
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
    envs = {"legacy": shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"])),
        "feasible": goals.ReacherFeasibleGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))}
    lo, hi = shared._normalized_action_bounds(mean, std, envs["legacy"].action_space.low, envs["legacy"].action_space.high)
    lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
    choices, pending, rows = [], [], []
    try:
        for source, old_choice in sources:
            seed, pair = source["seed"], {}
            for family, env in envs.items():
                context = shared._source_state_context(env, dataset=dataset, seed=seed)
                if any(context[k] != source[k] for k in ("source_episode", "source_start_row")):
                    raise RuntimeError("source identity changed")
                context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
                physics = env.unwrapped.env.physics
                effective, admission = goals.feasible_goal_delta(physics.data.qpos, ref["target_delta"], physics.model.jnt_limited, physics.model.jnt_range)
                pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
                    np.asarray(context["history_action_blocks"], np.float32), device)
                start = samples
                cost_fn = lambda p: p0._cost_fn(model, pixels, goal, history, p, action_lower=lower, action_upper=upper)
                plan, _, receipt = cem.search_with_incumbent(cost_fn, future_horizon=6, action_dim=history.shape[-1], device=device,
                    samples=128, steps=3, elite=16, seed=seed)
                actions = p0._to_raw_actions(plan, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]
                if samples - start != 2304:
                    raise RuntimeError("full CEM compute accounting changed")
                if family == "legacy" and (plan.tolist() != old_choice["cem_params"] or receipt != old_choice["cem_receipt"]
                        or context["goal_rgb_sha256"] != old_choice["goal_rgb_sha256"] or context["root_rgb_sha256"] != old_choice["root_rgb_sha256"]):
                    raise RuntimeError("historical context/search replay changed")
                record = {"seed": seed, "family": family, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                    "admission": admission, "target_qpos": np.asarray(env.unwrapped.env.task.target_qpos).tolist(),
                    "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                    "history_rgb_sha256": context["history_rgb_sha256"], "params": plan.tolist(), "receipt": receipt,
                    "action_witness_raw": actions.tolist()}
                choices.append(record)
                pair[family] = record
                pending.append((source, family, context, actions, record))
            if not pair["feasible"]["admission"]["flipped_joint_indices"]:
                if any(pair["legacy"][k] != pair["feasible"][k] for k in ("root_rgb_sha256", "goal_rgb_sha256", "history_rgb_sha256", "params", "receipt", "action_witness_raw")):
                    raise RuntimeError("valid-root input/plan must remain exactly unchanged")
            print(json.dumps({"seed": seed, "changed_task": bool(pair["feasible"]["admission"]["flipped_joint_indices"]),
                "legacy_target": pair["legacy"]["target_qpos"], "feasible_target": pair["feasible"]["target_qpos"]}), flush=True)
        write_new(paths["selection"], {"protocol_sha256": digest(paths["protocol"]), "rows": choices})
        selection_sha = digest(paths["selection"])
        print("All32 task-bound plans sealed; begin terminal evaluation.", flush=True)
        for source, _ in sources:
            seed, methods = source["seed"], {}
            for job_source, family, context, actions, record in pending:
                if job_source["seed"] != seed:
                    continue
                env = envs[family]
                frame, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                p0._verify_execution_identity(frame, goal, context["root_frame"], context["goal"])
                execution._execute_raw_steps(env, actions)
                physics = env.unwrapped.env.physics
                distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                prepared = processor._prepare_info(terminal_rgb_batch(env.render(), context["goal"], mean.size))
                with torch.no_grad():
                    rgb = float((p0._goal_latent(model, prepared["pixels"].to(device)) - p0._goal_latent(model, prepared["goal"].to(device))).square().mean())
                methods[family] = {"action_witness_raw": actions.tolist(), "physical": {"final_qpos_distance": distance, "final_rgb_cost": rgb}}
            changed = bool(next(record["admission"]["flipped_joint_indices"] for record in choices if record["seed"] == seed and record["family"] == "feasible"))
            gates = {"legacy_complete_replay_exact": methods["legacy"] == source["methods"]["cem_carry"],
                "valid_task_outcomes_unchanged": changed or methods["feasible"] == methods["legacy"],
                "all_bounded_finite": all(np.asarray(m["action_witness_raw"]).shape == (30, 2)
                    and np.isfinite(m["action_witness_raw"]).all() and np.abs(m["action_witness_raw"]).max() <= 1 + 1e-6
                    and np.isfinite(list(m["physical"].values())).all() for m in methods.values())}
            if not all(gates.values()):
                raise RuntimeError("target causal integrity failed: " + str(gates))
            row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                "changed_task": changed, "methods": methods, "gates": gates}
            rows.append(row)
            write_new(paths["roots"] / f"root_{seed}.json", row)
            print(json.dumps({"seed": seed, "changed_task": changed, "distances": {k: m["physical"]["final_qpos_distance"] for k, m in methods.items()}}), flush=True)
    finally:
        model.predict = predict
        for env in envs.values():
            env.close()
    gates = {"cuda_only": True, "all16_roots": [r["seed"] for r in rows] == protocol["root_seeds"],
        "all_exact_replays": all(all(r["gates"].values()) for r in rows), "weights_unchanged": before == shared._parameter_hash(model),
        "selection_seal_unchanged": digest(paths["selection"]) == selection_sha,
        "references_unchanged": all(digest(Path(r["path"])) == r["sha256"] for r in refs),
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    aggregate = {"changed_task_roots": [r["seed"] for r in rows if r["changed_task"]], "physical_branches": 32,
        "mean_distance": {k: float(np.mean([r["methods"][k]["physical"]["final_qpos_distance"] for r in rows])) for k in envs}}
    write_new(output, {"schema": "reacher_d2p_goal_joint_limits_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": selection_sha, "rows": rows,
        "gates": gates, "aggregate": aggregate, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
