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
NAME = "p3_reacher_feedback_cap_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P3_REACHER_FEEDBACK_CAP_20261101_CODEX.md"
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
    if (not np.array_equal(left["root_frame"], right["root_frame"])
            or not np.array_equal(left["goal"], right["goal"])
            or any(not np.array_equal(x, y) for x, y in zip(left["frames"], right["frames"], strict=True))
            or not np.array_equal(left["warmup_raw_steps"], right["warmup_raw_steps"])):
        raise ValueError("observed root/history mismatch")
    return dict(snapshot=a, root_rgb_sha256=left["root_rgb_sha256"],
                goal_rgb_sha256=left["goal_rgb_sha256"], history_rgb_sha256=left["history_rgb_sha256"],
                warmup_actions_sha256=left["warmup_actions_sha256"])


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
                 actions=[], chunks=[], checkpoints=[], error=None)
    try:
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
                    if len(packet_raw) == BLOCK:
                        actual_rgb = np.asarray(env.render(), np.uint8).copy()
                        ctx = update_context(ctx, root_before, actual_rgb, np.asarray(packet_raw))
                        root_before, packet_raw = actual_rgb, []
                        state["checkpoints"].append(dict(step=state["steps"],
                            pixel_sha256=hashlib.sha256(actual_rgb.tobytes()).hexdigest()))
                    return obs, reward, terminated, truncated, info

            phase[0] = "scheduling"
            outcome = adapter.run_chunk(model=model, env=ObservedEnv(), states=states,
                history_actions=history[:, 1:], action_tokens=torch.as_tensor(tokens, device=states.device,
                dtype=states.dtype)[None], physical_actions=raw, goal=goal,
                projection=torch.eye(states.shape[-1], device=states.device),
                token_steps=5, config=config, arm=arm, meter=meter)
            state["chunks"].append(dict(boundary=boundary,
                planning=receipt, candidate_raw=raw.tolist(), outcome=outcome,
                actual_returned=len(received), partial_packet=len(packet_raw)))
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
    sources = [Path(__file__).resolve(), DOC, STATS, Path(planned.__file__).resolve(),
               Path(adapter.__file__).resolve(), Path(kernel.__file__).resolve(),
               Path(context_api.__file__).resolve(), Path(locking.__file__).resolve(),
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
    model_before = model_after = None
    try:
        p0.install_missing_field_compat()
        contexts = []
        for arm in ("fixed", "dynamic"):
            env = p0.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
            envs.append(env)
            contexts.append(p0.collect_executed_context(env, seed=SEED, history_len=3, action_block=5))
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
            context_api.assert_exact_execution_state(ctx["execution_snapshot"],
                                                    context_api.capture_execution_snapshot(env))
            print(json.dumps(dict(arm=arm, status="started")), flush=True)
            rows.append(run_arm(arm, env, ctx, model, preprocessor, mean, std, p0, output))
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
            env.close()
    complete = failure is None and len(rows) == 2 and all(r["status"] == "completed15" for r in rows)
    result = dict(schema=NAME + ".report.v1", status="complete" if complete else "partial",
                  error=failure, rows=rows, root_audit_sha256=planned.file_hash(output / "root_audit.json")
                  if root_audit is not None else None,
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
