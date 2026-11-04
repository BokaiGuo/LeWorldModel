"""Host CUDA and physical replay; only executes with an explicit frozen plan and authority."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import reacher_p1_52_replay_contract_20261102_codexexclusive as contract
from . import reacher_p1_52_replay_roster_20261102_codexexclusive as roster
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host


def predict_fixed_tokens(model, pixels, history, normalized, p0, torch):
    """Mirror the causal rollout while passing executable normalized tokens unchanged."""
    host.require(tuple(normalized.shape) == (1, 2, 10)
                 and tuple(history.shape) == (1, 3, 10), "fixed token or history shape")
    encoded = p0._latent_context(model, pixels, history)
    action_context = torch.cat([history[:, 1:], normalized[:, :1]], dim=1)
    emb_context = encoded
    terminal = None
    for index in range(2):
        action_emb = model.action_encoder(action_context[:, -3:])
        terminal = model.predict(emb_context[:, -3:], action_emb)[:, -1]
        if index == 0:
            emb_context = torch.cat([emb_context, terminal[:, None]], dim=1)
            action_context = torch.cat([action_context, normalized[:, 1:2]], dim=1)
    host.require(terminal is not None and len(terminal.shape) == 2 and terminal.shape[0] == 1,
                 "fixed prediction output shape")
    return terminal


def replay_arm(row, root, env, setup, p0, processor, model, mean, std, lower, upper, journal, torch):
    key = {"group": row["group"], "seed": row["seed"], "arm": row["arm"]}
    actions = np.asarray(row["actions"], dtype=np.float32)
    host.require(host.bridge.sha_array(actions) == row["action_sha256"], "action digest")
    journal.emit({"kind": "arm_started", **key, "action_sha256": row["action_sha256"]})
    step_index = 0
    original_step = env.step
    warmup_index = 0

    def warmup_step(action):
        nonlocal warmup_index
        expected = np.asarray(root["warmup_packets"], dtype=np.float32).reshape(15, 2)
        host.require(warmup_index < 15 and np.array_equal(np.asarray(action, np.float32), expected[warmup_index]),
                     "warmup action mismatch")
        journal.emit({"kind": "warmup_intent", **key, "step": warmup_index,
                      "action": np.asarray(action, np.float32).tolist()})
        result = original_step(action)
        journal.emit({"kind": "warmup_returned", **key, "step": warmup_index,
                      "terminated": bool(result[2]), "truncated": bool(result[3])})
        warmup_index += 1
        return result

    def step(action):
        nonlocal step_index
        action = np.asarray(action, dtype=np.float32)
        host.require(step_index < 15 and np.array_equal(action, actions[step_index]), "action trace")
        journal.emit({"kind": "step_intent", **key, "step": step_index, "action": action.tolist()})
        result = original_step(action)
        journal.emit({"kind": "step_returned", **key, "step": step_index,
                      "terminated": bool(result[2]), "truncated": bool(result[3])})
        step_index += 1
        return result

    try:
        env.step = warmup_step
        context = setup.collect_executed_context(
            env, seed=row["seed"], history_len=3, action_block=5,
            warmup_packets=np.asarray(root["warmup_packets"], dtype=np.float32),
            warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
        host.require(warmup_index == 15, "warmup step count")
        host.bridge.check_context(context, root)
        root_frame, root_goal = setup.restore_execution_snapshot(
            env, seed=row["seed"], snapshot=context["execution_snapshot"])
        p0._verify_execution_identity(root_frame, root_goal, context["root_frame"], context["goal"])
        env.step = step
        cp5 = setup.advance_executed_context(env, context=context, raw_action_block=actions[:5],
                                             action_block=5, verified_root_frame=root_frame)
        cp5_value = host.capture(cp5["frames"], cp5["root_frame"], cp5["goal"],
                                 cp5["history_action_blocks"])
        host.require(roster.digest(cp5_value) == row["cp5_context_sha256"], "cp5 RGB/history drift")
        raw = env.unwrapped
        cp5_distance = p0._qpos_distance(np.asarray(raw.env.physics.data.qpos),
                                          np.asarray(raw.env.task.target_qpos))
        host.require(abs(cp5_distance - row["original_physical"]["after_first_chunk_qpos_distance"]) <= 1e-8,
                     "cp5 physical drift")
        journal.emit({"kind": "cp5_verified", **key, "context": cp5_value,
                      "physical_distance": float(cp5_distance)})
        pixels, goal, history = p0._prepare(processor, cp5["frames"], cp5["root_frame"], cp5["goal"],
                                             np.asarray(cp5["history_action_blocks"], np.float32), torch.device("cuda"))
        normalized = contract.normalized_suffix(actions, mean, std, lower.detach().cpu().numpy(),
                                                upper.detach().cpu().numpy())
        journal.emit({"kind": "model_intent", **key, "phase": "prediction",
                      "suffix_normalized": normalized.tolist(), "action_sha256": row["action_sha256"]})
        with torch.no_grad():
            predicted = predict_fixed_tokens(model, pixels, history,
                                             torch.as_tensor(normalized[None], device="cuda"),
                                             p0, torch)
            predicted = p0._planning_coordinate(model, predicted[0], role="predicted")
            goal_latent = p0._goal_latent(model, goal)
            target = p0._planning_coordinate(model, goal_latent, role="goal")
        journal.emit({"kind": "model_returned", **key, "phase": "prediction",
                      "predicted": predicted.detach().cpu().tolist(), "goal": target.detach().cpu().tolist()})
        restored, restored_goal = setup.restore_execution_snapshot(
            env, seed=row["seed"], snapshot=cp5["execution_snapshot"])
        p0._verify_execution_identity(restored, restored_goal, cp5["root_frame"], cp5["goal"])
        middle = setup.advance_executed_context(env, context=cp5, raw_action_block=actions[5:10],
                                                action_block=5, verified_root_frame=restored)
        final = setup.advance_executed_context(env, context=middle, raw_action_block=actions[10:],
                                               action_block=5)
        host.require(step_index == 15, "physical step count")
        physical = p0._qpos_distance(np.asarray(raw.env.physics.data.qpos),
                                     np.asarray(raw.env.task.target_qpos))
        final_value = host.capture(final["frames"], final["root_frame"], final["goal"],
                                   final["history_action_blocks"])
        journal.emit({"kind": "physical_returned", **key, "context": final_value,
                      "physical_distance": float(physical),
                      "original_final": row["original_physical"]["final_qpos_distance"]})
        host.require(abs(physical - row["original_physical"]["final_qpos_distance"]) <= 1e-8,
                     "final physical drift")
        final_pixels, _, final_history = p0._prepare(
            processor, final["frames"], final["root_frame"], final["goal"],
            np.asarray(final["history_action_blocks"], np.float32), torch.device("cuda"))
        journal.emit({"kind": "model_intent", **key, "phase": "actual_rgb_encode"})
        with torch.no_grad():
            actual_latent = p0._latent_context(model, final_pixels, final_history)[0, -1]
            actual = p0._planning_coordinate(model, actual_latent, role="observed")
        vectors = {"predicted": predicted.detach().cpu().tolist(),
                   "actual": actual.detach().cpu().tolist(), "goal": target.detach().cpu().tolist()}
        values = contract.metrics(**vectors, physical=float(physical))
        journal.emit({"kind": "model_returned", **key, "phase": "actual_rgb_encode",
                      "actual": vectors["actual"]})
        journal.emit({"kind": "arm_complete", **key, "metrics": values, "steps": step_index})
    except Exception as exc:
        journal.emit({"kind": "arm_failed", **key, "returned_steps": step_index,
                      "error_type": type(exc).__name__, "error": str(exc)[:1000]})
    finally:
        env.step = original_step


def run(plan_path, expected, output, authority):
    host.require(authority == contract.AUTHORITY and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID,
                 "coordinated authority and CUDA UUID required")
    plan = contract.bind_plan(plan_path, expected)
    output = Path(output)
    host.require(output.is_absolute() and output.parent == contract.ROOT / "results"
                 and output.name.startswith("reacher_p1_52_replay_RUN_") and not os.path.lexists(output),
                 "new exclusive output directory")
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        import torch
        from experiments.gpu_preflight import require_cuda_allocation
        from . import preflight_reacher_d2p_p1_reconfiguration as setup
        from . import run_reacher_d2p_p0_goal_progress as p0

        output.mkdir(mode=0o700)
        host.write_new(output / "plan.json", plan)
        journal = host.durable.Journal(output / "events.jsonl")
        env = None
        try:
            journal.emit({"kind": "run_started", "plan_sha256": expected})
            allocation = require_cuda_allocation("cuda:0")
            host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
                torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "CUDA identity")
            journal.emit({"kind": "cuda_ready", "allocation": allocation, "gpu_uuid": host.UUID})
            setup.install_missing_field_compat()
            setup.validate_training_semantic_receipt(Path(plan["stats"]))
            mean, std = setup._load_action_stats(Path(plan["stats"]), "swm/ReacherDMControl-v0")
            model = setup.load_compatible_pretrained(Path(plan["weights"]).parent, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            host.require(before == plan["model_parameter_hash"], "model hash")
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            env = setup.ReacherGoalImageWrapper(
                gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
            p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
            roots = {r["seed"]: r for r in plan["roots"]}
            rows = json.loads(Path(plan["roster"]).read_text())["rows"]
            for row in rows:
                replay_arm(row, roots[row["seed"]], env, setup, p0, processor, model,
                           mean, std, lower, upper, journal, torch)
            after = setup._parameter_hash(model)
            host.require(before == after and contract.bind_plan(plan_path, expected) == plan, "postcheck drift")
            journal.emit({"kind": "run_finished", "model_before": before, "model_after": after})
        except Exception as exc:
            journal.emit({"kind": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            if env is not None:
                env.close()
            journal.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    for name in ("plan", "sha256", "output", "authority"):
        execute.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.output, contract.prepare())
        print(json.dumps({"plan": args.output, "sha256": contract.sha(args.output)}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
