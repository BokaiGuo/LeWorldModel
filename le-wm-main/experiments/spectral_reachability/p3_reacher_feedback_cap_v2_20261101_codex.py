"""Real Reacher feedback pair with exact audited root and common model-entry caps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import codex_p3_reacher_planned_exclusive_20261101 as planned
from . import codex_p3_live_chunk_adapter_20261101 as adapter
from . import codex_p3_chunk_extension_20261101 as kernel
from . import reacher_d2p_context as context_api
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

ROOT = Path(__file__).resolve().parents[2]
NAME = "p3_reacher_feedback_cap_v2_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P3_REACHER_FEEDBACK_CAP_V2_20261101_CODEX.md"
POLICY = Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm")
STATS = ROOT / "results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
CAPS = {"predict_calls": 48, "predict_samples": 192, "encoder_calls": 48, "encoder_samples": 192,
        "action_encoder_calls": 64, "action_encoder_samples": 256}
TOTAL, BLOCK, SEED = 15, 5, 20261101


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Budget:
    def __init__(self, caps):
        if set(caps) != set(CAPS) or any(type(v) is not int or v < 0 for v in caps.values()):
            raise ValueError("complete finite caps")
        self.caps = dict(caps)
        self.attempted = dict.fromkeys(caps, 0)
        self.events = []
        self.failed = False

    def charge(self, kind, samples, phase):
        if self.failed or kind not in ("predict", "encoder", "action_encoder") or type(samples) is not int or samples < 1:
            raise ValueError("inactive budget or invalid model entry")
        count, n = kind + "_calls", kind + "_samples"
        if self.attempted[count] + 1 > self.caps[count] or self.attempted[n] + samples > self.caps[n]:
            self.failed = True
            self.events.append(dict(phase=phase, kind=kind, samples=samples, status="blocked"))
            raise ValueError("common cap exceeded before dispatch")
        self.attempted[count] += 1
        self.attempted[n] += samples
        self.events.append(dict(phase=phase, kind=kind, samples=samples, status="attempted"))


def audit_roots(left, right):
    a, b = left["execution_snapshot"], right["execution_snapshot"]
    required = ("physics_integration_state", "integration_state_spec", "model_binary_sha256",
                "target_qpos", "target_marker_pos", "rendered_goal_rgb")
    if a is None or b is None or any(k not in a or k not in b for k in required):
        raise ValueError("complete integration/scene root unavailable")
    if any(a[k] != b[k] for k in required):
        raise ValueError("physical integration or target mismatch")
    if a != b:
        raise ValueError("complete saved root snapshots differ")
    if (not np.array_equal(left["root_frame"], right["root_frame"])
            or not np.array_equal(left["goal"], right["goal"])
            or any(not np.array_equal(x, y) for x, y in zip(left["frames"], right["frames"], strict=True))
            or not np.array_equal(left["warmup_raw_steps"], right["warmup_raw_steps"])):
        raise ValueError("observed root/history mismatch")
    return dict(snapshot=a, root_rgb_sha256=left["root_rgb_sha256"],
                goal_rgb_sha256=left["goal_rgb_sha256"], history_rgb_sha256=left["history_rgb_sha256"],
                warmup_actions_sha256=left["warmup_actions_sha256"])


def physical_geometry(env):
    """Evaluator-only qpos geometry; do not invent success for the hard task."""
    from dm_control.suite import reacher as dm_reacher
    from stable_worldmodel.envs.dmcontrol.custom_tasks.reacher import ReacherQPosMatchTask

    raw = env.unwrapped
    task, physics = raw.env.task, raw.env.physics
    is_hard = raw._task_name == "hard" and type(task) is dm_reacher.Reacher
    is_qpos = raw._task_name == "qpos_match" and type(task) is ReacherQPosMatchTask
    if not (is_hard or is_qpos):
        raise ValueError("unrecognized native Reacher task")
    qpos = np.asarray(physics.data.qpos, dtype=np.float64).copy()
    target = np.asarray(task.target_qpos, dtype=np.float64).copy()
    if (qpos.ndim != 1 or not qpos.size or qpos.shape != target.shape
            or not np.isfinite(qpos).all() or not np.isfinite(target).all()):
        raise ValueError("invalid native qpos geometry")
    difference = qpos - target
    common = dict(qpos=qpos.tolist(), target_qpos=target.tolist(),
                  qpos_difference=difference.tolist(),
                  qpos_l2=float(np.linalg.norm(difference)),
                  qpos_max_abs=float(np.max(np.abs(difference))))
    if is_hard:
        return dict(**common, native_success=None, native_termination=None,
                    native_qpos_threshold=None,
                    criterion="unavailable: hard task has no native qpos success criterion")
    threshold = float(task.qpos_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid native qpos threshold")
    native = task.get_termination(physics)
    if native is not None and float(native) != 0.0:
        raise ValueError("unexpected native termination return")
    matched = bool(np.all(np.abs(difference) < threshold))
    if matched != (native is not None):
        raise ValueError("native termination does not match saved geometry")
    return dict(**common, native_qpos_threshold=threshold, native_termination=native,
                native_success=matched,
                criterion="all(abs(qpos-target_qpos) < qpos_threshold), no angle wrapping")


def update_context(ctx, root_before, next_frame, packet):
    frames, blocks = list(ctx["frames"]), list(ctx["history_action_blocks"])
    if len(frames) != 3 or len(blocks) != 3 or np.asarray(packet).shape != (5, 2):
        raise ValueError("actual packet/history shape")
    result = dict(ctx)
    result["frames"] = frames[1:] + [np.asarray(root_before, np.uint8).copy()]
    result["history_action_blocks"] = blocks[1:] + [np.asarray(packet, np.float32).reshape(-1).copy()]
    result["root_frame"] = np.asarray(next_frame, np.uint8).copy()
    return result


def run_arm(arm, env, ctx, model, preprocessor, mean, std, p0, output, *, device="cuda"):
    budget = Budget(CAPS)
    phase = ["setup"]
    hooks = []
    original_predict = model.predict
    def predict(states, actions, *args, **kwargs):
        budget.charge("predict", int(states.shape[0]), phase[0])
        return original_predict(states, actions, *args, **kwargs)
    model.predict = predict
    for name in ("encoder", "action_encoder"):
        def before(module, inputs, name=name):
            budget.charge(name, int(inputs[0].shape[0]), phase[0])
        hooks.append(getattr(model, name).register_forward_pre_hook(before))
    state = dict(arm=arm, status="failed", steps=0, physical_step_attempts=0,
                 actions=[], chunks=[], checkpoints=[], error=None, root_geometry=None)
    try:
        state["root_geometry"] = physical_geometry(env)
        for index in range(3):
            remaining = TOTAL - state["steps"]
            if not remaining:
                break
            pixels, goal_rgb, history = p0._prepare(
                preprocessor, ctx["frames"], ctx["root_frame"], ctx["goal"],
                np.asarray(ctx["history_action_blocks"], np.float32), torch.device(device))
            phase[0] = "context"
            states = p0._latent_context(model, pixels, history)
            goal = p0._goal_latent(model, goal_rgb)
            phase[0] = "planning"
            raw, receipt = planned.plan_actions(
                p0, model, pixels, goal_rgb, history, mean, std,
                env.action_space.low, env.action_space.high, SEED + 2000 + index,
                samples=8, steps=2, elite=2)
            tokens, raw = planned.bind_actions(raw, mean, std, env.action_space.low, env.action_space.high, BLOCK)
            config = kernel.Config(5, 5, min(2, remaining // 5 - 1), .01, 1.)
            meter = kernel.Meter({axis: 8 if axis in ("predict", "predictor_samples") else 0 for axis in kernel.AXES})
            root_before, packet_raw = ctx["root_frame"].copy(), []
            received = []
            boundary = state["steps"]

            class ObservedEnv:
                def step(self, action):
                    nonlocal ctx, root_before, packet_raw
                    action = np.asarray(action, np.float32).copy()
                    state["physical_step_attempts"] += 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(reward):
                        raise ValueError("nonfinite physical reward")
                    packet_raw.append(action)
                    received.append(dict(action=action.tolist(), reward=float(reward),
                                         terminated=bool(terminated), truncated=bool(truncated)))
                    state["actions"].append(received[-1])
                    state["steps"] += 1
                    received[-1]["geometry"] = physical_geometry(env)
                    if len(packet_raw) == BLOCK:
                        actual_rgb = np.asarray(env.render(), np.uint8).copy()
                        ctx = update_context(ctx, root_before, actual_rgb, np.asarray(packet_raw))
                        root_before, packet_raw = actual_rgb, []
                        state["checkpoints"].append(dict(step=state["steps"],
                            pixel_sha256=hashlib.sha256(actual_rgb.tobytes()).hexdigest()))
                    return obs, reward, terminated, truncated, info

            phase[0] = "scheduling"
            chunk = dict(boundary=boundary, planning=receipt, candidate_raw=raw.tolist(),
                         outcome=None, actual_returned=0, partial_packet=0, status="started")
            state["chunks"].append(chunk)
            try:
                outcome = adapter.run_chunk(model=model, env=ObservedEnv(), states=states,
                    history_actions=history[:, 1:], action_tokens=torch.as_tensor(tokens, device=states.device,
                    dtype=states.dtype)[None], physical_actions=raw, goal=goal,
                    projection=torch.eye(states.shape[-1], device=states.device),
                    token_steps=5, config=config, arm=arm, meter=meter)
                chunk.update(outcome=outcome, status="returned")
            finally:
                chunk.update(actual_returned=len(received), partial_packet=len(packet_raw))
            print(json.dumps(dict(arm=arm, steps=state["steps"], selected=outcome["chunk_steps"])), flush=True)
            if outcome["terminated"] or outcome["truncated"]:
                state["status"] = "terminated" if outcome["terminated"] else "truncated"
                break
            if packet_raw:
                raise ValueError("partial nonterminal packet")
        else:
            if state["steps"] != TOTAL:
                raise ValueError("decision count exhausted")
        if state["steps"] == TOTAL and state["status"] == "failed":
            state["status"] = "completed15"
    except Exception as exc:
        state["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        model.predict = original_predict
        for hook in hooks:
            hook.remove()
        state["budget"] = dict(caps=budget.caps, attempted=budget.attempted, events=budget.events)
        planned.write_once(output / (arm + ".json"), state)
    return state


def run_locked(operation, *, lock_path=None):
    """Hold the shared advisory lock across the whole experiment callback."""
    with locking.gpu_lock(LOCK if lock_path is None else lock_path):
        return operation()


def run_experiment(args):
    output = Path(args.output)
    if (not output.is_absolute() or output.parent != ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    import inspect
    from stable_worldmodel.envs.dmcontrol import reacher as native_wrapper
    from stable_worldmodel.envs.dmcontrol.custom_tasks import reacher as native_task
    from stable_worldmodel.envs.dmcontrol import dmcontrol as native_base
    from dm_control.suite import reacher as dm_reacher
    sources = [Path(__file__).resolve(), DOC, STATS, Path(planned.__file__).resolve(),
               Path(adapter.__file__).resolve(), Path(kernel.__file__).resolve(),
               Path(context_api.__file__).resolve(), Path(locking.__file__).resolve(),
               Path(inspect.getfile(native_wrapper)).resolve(), Path(inspect.getfile(native_task)).resolve(),
               Path(inspect.getfile(native_base)).resolve(),
               Path(inspect.getfile(dm_reacher)).resolve(),
               ROOT / "experiments/run_live_reacher_planning.py",
               *sorted(p for p in POLICY.rglob("*") if p.is_file())]
    fingerprints = {str(p): planned.file_hash(p) for p in sources}
    output.mkdir()
    planned.write_once(output / "protocol.json", dict(schema=NAME + ".protocol.v1",
        arms=["fixed", "dynamic"], total_steps=TOTAL, packet_steps=BLOCK, seed=SEED,
        caps=CAPS, cem=dict(samples=8, steps=2, elite=2),
        min_improvement=.01, max_disagreement=1., sources=fingerprints,
        cooperative_lock=str(LOCK),
        scope="known root development, common entry cap not equal compute, P1/P3 not admitted"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
    from experiments.gpu_preflight import require_cuda_allocation
    import gymnasium as gym
    envs, rows, failure, root_audit = [], [], None, None
    snapshots = []
    model_before = model_after = None
    try:
        p0.install_missing_field_compat()
        contexts = []
        for arm in ("fixed", "dynamic"):
            env = p0.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
            envs.append(env)
            contexts.append(p0.collect_executed_context(env, seed=SEED, history_len=3, action_block=5))
            ctx = contexts[-1]
            snapshots.append(dict(arm=arm, snapshot=ctx["execution_snapshot"],
                root_rgb_sha256=ctx["root_rgb_sha256"], goal_rgb_sha256=ctx["goal_rgb_sha256"],
                history_rgb_sha256=ctx["history_rgb_sha256"],
                warmup_actions_sha256=ctx["warmup_actions_sha256"]))
            planned.write_once(output / ("root_snapshot_" + arm + ".json"), snapshots[-1])
        planned.write_once(output / "root_snapshots.json", dict(rows=snapshots))
        root_audit = audit_roots(contexts[0], contexts[1])
        planned.write_once(output / "root_audit.json", root_audit)
        runtime_sources = planned.loaded_project_sources()
        planned.write_once(output / "runtime_manifest.json", dict(loaded_project_python_sha256=runtime_sources,
            scope="modules loaded before model creation; late imports and third-party modules not exhaustively sealed"))
        require_cuda_allocation("cuda")
        p0.validate_training_semantic_receipt(STATS)
        mean, std = p0._load_action_stats(STATS, "swm/ReacherDMControl-v0")
        model = p0.load_compatible_pretrained(str(POLICY), device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        model_before = model_digest(model)
        p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        preprocessor = p0._Preprocessor(p0.ActionNormalizer(mean, std))
        for arm, env, ctx in zip(("fixed", "dynamic"), envs, contexts, strict=True):
            try:
                context_api.assert_exact_execution_state(ctx["execution_snapshot"],
                                                        context_api.capture_execution_snapshot(env))
                print(json.dumps(dict(arm=arm, status="started")), flush=True)
                rows.append(run_arm(arm, env, ctx, model, preprocessor, mean, std, p0, output))
            except Exception as exc:
                rows.append(dict(arm=arm, status="failed", steps=0, physical_step_attempts=0,
                    actions=[], chunks=[], checkpoints=[], root_geometry=None,
                    error=dict(type=type(exc).__name__, message=str(exc)[:1000])))
            print(json.dumps(dict(arm=arm, status=rows[-1]["status"], steps=rows[-1]["steps"])), flush=True)
        torch.cuda.synchronize()
        model_after = model_digest(model)
        if model_before != model_after:
            raise ValueError("model state changed")
        if any(planned.file_hash(p) != h for p, h in (fingerprints | runtime_sources).items()):
            raise ValueError("source drift")
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                if failure is None:
                    failure = dict(type=type(exc).__name__, message="environment close: " + str(exc)[:950])
    for arm in ("fixed", "dynamic"):
        if not any(row["arm"] == arm for row in rows):
            rows.append(dict(arm=arm, status="not_started", steps=0, physical_step_attempts=0,
                actions=[], chunks=[], checkpoints=[], root_geometry=None,
                error=failure or dict(type="NotStarted", message="upstream failure")))
    complete = failure is None and len(rows) == 2 and all(r["status"] == "completed15" for r in rows)
    result = dict(schema=NAME + ".report.v1", status="complete" if complete else "partial",
                  error=failure, rows=rows, root_audit_sha256=planned.file_hash(output / "root_audit.json")
                  if root_audit is not None else None,
                  root_snapshots_sha256=planned.file_hash(output / "root_snapshots.json")
                  if (output / "root_snapshots.json").exists() else None,
                  individual_root_sha256={arm: planned.file_hash(output / ("root_snapshot_" + arm + ".json"))
                      for arm in ("fixed", "dynamic")
                      if (output / ("root_snapshot_" + arm + ".json")).exists()},
                  actual_same_horizon=complete, common_caps=CAPS, equal_actual_compute=False,
                  model_before=model_before, model_after=model_after,
                  P1_admission=False, P3_admission=False,
                  scope="single known root, model-entry caps only, no independent confirmation")
    planned.write_once(output / "report.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not complete:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda-and-physical", required=True, action="store_true")
    args = parser.parse_args()
    run_locked(lambda: run_experiment(args))


if __name__ == "__main__":
    main()
