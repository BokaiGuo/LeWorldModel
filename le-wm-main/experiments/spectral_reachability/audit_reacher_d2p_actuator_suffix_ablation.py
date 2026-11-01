"""CUDA-gated physical witness replay and suffix-zero ablation of task pilot.

Every stored 15-step program is replayed without replanning. A paired control
keeps the exact first five raw actions and replaces only its suffix by ten
literal zero actions. This is evaluator-only causal diagnosis on known roots,
not action selection, new efficacy, or a controllability theorem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_actuator_limit_tasks as tasks
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_frame(env):
    # MuJoCo may expose a vertically flipped, negative-stride view. Keep
    # pixel values/orientation unchanged while making it tensor-compatible.
    return np.ascontiguousarray(env.render(), dtype=np.uint8)


def suffix_effect_metrics(methods):
    effects = {}
    for name, method in methods.items():
        optimized, passive = method["optimized"], method["prefix_then_zero_suffix"]
        effects[name] = {"qpos_suffix_improvement": passive["final_qpos_distance"] - optimized["final_qpos_distance"],
            "rgb_suffix_improvement": passive["observed_rgb_goal_cost"] - optimized["observed_rgb_goal_cost"]}
    return {"effects": effects,
        "dispatch_minus_greedy_suffix_improvement": effects["dispatch"]["qpos_suffix_improvement"] - effects["same_bank_greedy"]["qpos_suffix_improvement"],
        "dispatch_passive_minus_greedy_passive": methods["dispatch"]["prefix_then_zero_suffix"]["final_qpos_distance"] - methods["same_bank_greedy"]["prefix_then_zero_suffix"]["final_qpos_distance"],
        "dispatch_optimized_minus_greedy_optimized": methods["dispatch"]["optimized"]["final_qpos_distance"] - methods["same_bank_greedy"]["optimized"]["final_qpos_distance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    if report.get("schema") != "reacher_d2p_actuator_limit_task_pilot.v1" or report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("complete integrity-passing task pilot required")
    reference = report["sealed_protocol"]
    expected_cells = {(seed, task) for seed in reference["root_seeds"] for task in reference["tasks"]}
    cells = [(row["seed"], row["task"]) for row in report["rows"]]
    if len(cells) != len(set(cells)) or set(cells) != expected_cells:
        raise ValueError("complete unique root/task cells required")
    if digest(args.policy / "weights.pt") != reference["weights_sha256"] or digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("reference weights/normalizer differ")
    modules = {"context": Path(execution.__file__), "p0": Path(p0.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    if any(digest(path) != reference["implementation_sha256"][key] for key, path in modules.items()):
        raise ValueError("execution implementation changed since task pilot")
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".cells"))):
        raise FileExistsError("new output and sidecars required")
    sealed = {"root_seeds": reference["root_seeds"], "tasks": reference["tasks"], "reference_sha256": digest(args.reference),
        "weights_sha256": reference["weights_sha256"], "adapter_sha256": reference["adapter_sha256"],
        "source_sha256": reference["source_sha256"], "auditor_sha256": digest(Path(__file__)),
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "intervention": "exact stored 5-step prefix plus 10 literal raw-zero steps; compare with full stored 15-step witness",
        "final_qpos_replay_identity": "exact", "scope": "Known-root physical causal diagnostic, no selection or efficacy admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".cells").mkdir()
    print("Replaying every actual task program and its prefix-fixed, zero-suffix intervention; neural RGB scores on host CUDA only.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_hash = shared.receipt_source(args.adapter_artifact)
    if source_hash != reference["source_sha256"]:
        raise ValueError("source receipt changed")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["goal_delta"]))
    rows = []
    try:
        for source in sorted(report["rows"], key=lambda row: (row["seed"], row["task"])):
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")):
                raise RuntimeError("reference source root/goal changed")
            _, goal, _ = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
            goal_latent = p0._goal_latent(model, goal)
            methods = {}
            for name in ("dispatch", "same_bank_greedy", "zero"):
                original = source["methods"][name]
                raw = np.asarray(original["action_witness_raw"], np.float32)
                passive_raw = np.concatenate((raw[:5], np.zeros((10, 2), np.float32)), axis=0)
                method = {}
                for condition, actions in (("optimized", raw), ("prefix_then_zero_suffix", passive_raw)):
                    if not tasks.action_limit_gate({"action_witness_raw": actions}, source["raw_torque_limits"]):
                        raise RuntimeError("intervention violates physical task bounds")
                    physical, identity = p0._execute(env, seed, actions, expected_frame=context["root_frame"],
                        expected_goal=context["goal"], execution_snapshot=context["execution_snapshot"])
                    frame = render_frame(env)
                    prepared = processor._prepare_info({"pixels": frame[None, None], "goal": context["goal"][None, None], "action": np.zeros((1, 1, mean.size), np.float32)})
                    latent = p0._goal_latent(model, prepared["pixels"].to(device))
                    method[condition] = {"final_qpos_distance": physical["final_qpos_distance"],
                        "observed_rgb_goal_cost": float((latent - goal_latent).square().mean()), "execution_identity": identity,
                        "raw_action_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()}
                method["optimized_reference_final_qpos_error"] = abs(method["optimized"]["final_qpos_distance"] - original["physical"]["final_qpos_distance"])
                methods[name] = method
            metrics = suffix_effect_metrics(methods)
            row = {"seed": seed, "task": source["task"], "source_episode": source["source_episode"],
                "model_eligible": source["methods"]["dispatch"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"],
                "methods": methods, "metrics": metrics,
                "gates": {"all_full_witness_final_qpos_replays_exactly": all(method["optimized_reference_final_qpos_error"] == 0 for method in methods.values())}}
            with (output.with_suffix(".cells") / f"{source['task']}_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("seed", "task", "model_eligible", "gates", "metrics")}), flush=True)
    finally:
        env.close()
    aggregate = {}
    for task in reference["tasks"]:
        subset = [row for row in rows if row["task"] == task]
        eligible = [row for row in subset if row["model_eligible"]]
        aggregate[task] = {"root_count": len(subset), "eligible_count": len(eligible),
            "eligible_stronger_suffix_effect_than_greedy": sum(row["metrics"]["dispatch_minus_greedy_suffix_improvement"] > 0 for row in eligible),
            "eligible_passive_worse_but_optimized_better_than_greedy": sum(row["metrics"]["dispatch_passive_minus_greedy_passive"] > 0 and row["metrics"]["dispatch_optimized_minus_greedy_optimized"] < 0 for row in eligible),
            "eligible_metrics": [{"seed": row["seed"], **row["metrics"]} for row in eligible]}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_cells_retained": len(rows) == len(expected_cells) and {(row["seed"], row["task"]) for row in rows} == expected_cells,
        "all_full_witnesses_replay_exactly": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == sealed["implementation_sha256"][key] for key, path in modules.items()) and digest(Path(__file__)) == sealed["auditor_sha256"]}
    result = {"schema": "reacher_d2p_actuator_suffix_ablation.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": sealed, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior full-source SHA receipt reused; all source root/goal identities matched exactly. No new full-dataset SHA claimed.",
        "claim_boundary": "Same-root evaluator intervention. Larger control effects alone are not controllability change, efficacy, or formal P1 admission."}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
