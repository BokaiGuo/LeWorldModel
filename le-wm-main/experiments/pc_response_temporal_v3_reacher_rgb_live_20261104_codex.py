"""Run the same v3 compiler on a frozen Reacher model with actual RGB feedback."""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_response_temporal_v3_reacher_adapter_preflight_v2_20261104_codex as prior
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import reacher_d2p_context as context_lib

run = prior.run
host, setup, p0, adapter, method, ROOT = run.host, run.setup, run.p0, run.adapter, run.method, run.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_REACHER_RGB_LIVE_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v3_reacher_rgb_live_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
ARMS = ("response", "smooth_cem")
PRE_MANIFEST = prior.PREFIX.with_name(prior.PREFIX.name + "_manifest.json")
PRE_CONTEXT = prior.PREFIX.with_name(prior.PREFIX.name + "_context.npz")
PRE_RESULT = prior.PREFIX.with_name(prior.PREFIX.name + "_result.json")


def path(arm, suffix):
    return PREFIX.with_name(PREFIX.name + "_" + arm + "_" + suffix)


def dependencies(source):
    paths = (Path(__file__), PROTOCOL, PRE_RESULT, PRE_MANIFEST, PRE_CONTEXT, prior.PROTOCOL,
             Path(prior.__file__), Path(context_lib.__file__))
    return {**run.dependencies(source), **{str(p.resolve()): host.bridge.sha_file(p) for p in paths}}


def save_context(filename, context):
    with filename.open("xb") as stream:
        np.savez_compressed(stream, frames=np.stack(context["frames"]), root=context["root_frame"],
                            goal=context["goal"], history=np.asarray(context["history_action_blocks"], np.float32))
        stream.flush()
        os.fsync(stream.fileno())


def physical(env):
    inner = env.unwrapped.env
    qpos, goal = np.asarray(inner.physics.data.qpos).copy(), np.asarray(inner.task.target_qpos).copy()
    return {"qpos": qpos.tolist(), "target_qpos": goal.tolist(),
            "distance": float(p0._qpos_distance(qpos, goal))}


def main():
    all_paths = [MANIFEST] + [path(arm, f"context{step}.npz") for arm in ARMS for step in range(4)]
    all_paths += [path(arm, f"decision{step}.json") for arm in ARMS for step in range(3)]
    all_paths += [path(arm, "execution.json") for arm in ARMS]
    if any(p.exists() for p in all_paths):
        raise FileExistsError("existing or partial live experiment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    source = json.loads(run.SOURCE.read_text())
    receipt = json.loads(PRE_RESULT.read_text())
    sha = host.bridge.sha_file
    if (receipt["manifest_sha256"] != sha(PRE_MANIFEST) or receipt["context_sha256"] != sha(PRE_CONTEXT)
            or receipt["model_parameter_hash"] != source["model_parameter_hash_expected"]):
        raise ValueError("preflight seal mismatch")
    hashes = dependencies(source)
    host.write_new(MANIFEST, {"dependencies": hashes, "method": method.METHOD, "root": 20263160,
                              "method_root_index": 0, "arms": ARMS, "budget": 1920,
                              "scope": "one exposed root; RGB live development, not confirmation"})
    print("Reacher v3 live RGB: two arms, one exposed root, fixed 15-step terminal.", flush=True)
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        from experiments.gpu_preflight import require_cuda_allocation

        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        setup.install_missing_field_compat()
        stats = Path(source["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
        model = setup.load_compatible_pretrained(Path(source["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        model_hash = setup._parameter_hash(model)
        if model_hash != source["model_parameter_hash_expected"]:
            raise ValueError("frozen parameter mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.65, -.65))
        try:
            if (not np.array_equal(env.action_space.low, np.full(2, -1))
                    or not np.array_equal(env.action_space.high, np.full(2, 1))):
                raise ValueError("raw action box mismatch")
            p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            lo, hi = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
            for arm in ARMS:
                packets, history_index = setup.sample_history_packets(histories, seed=20263160)
                context = setup.collect_executed_context(env, seed=20263160, history_len=3, action_block=5,
                    warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                if (int(history_index) != receipt["history_index"]
                        or context["root_rgb_sha256"] != receipt["root_sha256"]
                        or context["goal_rgb_sha256"] != receipt["goal_sha256"]):
                    raise ValueError("reset identity mismatch")
                carry, rows = None, []
                outcome = [physical(env)]
                for step in range(3):
                    context_file = path(arm, f"context{step}.npz")
                    save_context(context_file, context)
                    encoded_at = time.monotonic()
                    pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"],
                        context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
                    with torch.no_grad():
                        encoded, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                        chart = adapter.goal_chart(encoded, goal_latent, 2027111800 + step)
                        target = (goal_latent @ chart).cpu().numpy()
                        torch.cuda.synchronize()
                    encoding_seconds = time.monotonic() - encoded_at
                    meter, trace = Meter(1920), []

                    def predict(actions):
                        requested = np.asarray(actions, dtype=float)
                        raw = requested.astype(np.float32)

                        def evaluate(batch):
                            future = adapter.raw_to_packets(batch, mean, std, device=device)
                            with torch.no_grad():
                                terminal = adapter.cached_terminal(model, encoded, history, future, lower, upper)
                                projected = terminal @ chart
                                scores = (terminal - goal_latent).square().mean(dim=1)
                                torch.cuda.synchronize()
                            return projected.cpu().numpy(), scores.cpu().numpy()

                        positions, scores = meter.predict(raw, evaluate)
                        trace.append({"requested": requested.tolist(), "raw": raw.tolist(),
                                      "positions": positions.tolist(), "scores": scores.tolist(),
                                      "tokens": len(raw) * raw.shape[1] // 5})
                        return positions, scores

                    started = time.monotonic()
                    result = method.decide(arm, 0, step, target, predict, meter, carry)
                    selected = np.asarray(result["selected"]["chain"], dtype=np.float32)
                    if selected.shape != (15 - 5 * step, 2) or meter.token_samples > 1920:
                        raise ValueError("invalid output action or budget")
                    decision_file = path(arm, f"decision{step}.json")
                    host.write_new(decision_file, {"manifest_sha256": sha(MANIFEST), "context_sha256": sha(context_file),
                        "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                        "history_rgb_sha256": context["history_rgb_sha256"], "carry_in": carry,
                        "chart_seed": 2027111800 + step, "chart": chart.cpu().tolist(), "target": target.tolist(),
                        "result": result, "trace": trace, "counts": meter.as_dict(),
                        "dispatch_float32": selected[:5].tolist(), "encoding_calls": 2,
                        "encoding_seconds": encoding_seconds, "planning_seconds": time.monotonic() - started})
                    seal_ns = time.monotonic_ns()
                    advanced = context_lib.advance_executed_context(env, context=context, raw_action_block=selected[:5],
                                                                     action_block=5, verified_root_frame=context["root_frame"])
                    execute_ns = time.monotonic_ns()
                    outcome.append(physical(env))
                    rows.append({"step": step, "decision_sha256": sha(decision_file),
                                 "context_sha256": sha(context_file), "sealed_ns": seal_ns,
                                 "execution_return_ns": execute_ns, "dispatch": selected[:5].tolist(),
                                 "successor_root_sha256": advanced["root_rgb_sha256"],
                                 "advanced_action_sha256": advanced["advanced_action_sha256"]})
                    carry, context = (selected[5:].tolist() if step < 2 else None), advanced
                    print(f"{arm} step={step}: {meter.token_samples} tokens; 5 actions executed.", flush=True)
                final_context = path(arm, "context3.npz")
                save_context(final_context, context)
                host.write_new(path(arm, "execution.json"), {"arm": arm, "manifest_sha256": sha(MANIFEST),
                    "rows": rows, "initial_and_block_outcomes": outcome, "final_context_sha256": sha(final_context),
                    "model_hash": model_hash, "cuda_allocation": allocation,
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
                    "full_cost_certified": False})
            if setup._parameter_hash(model) != model_hash or dependencies(source) != hashes:
                raise ValueError("model or dependency drift")
        finally:
            env.close()
    print("Reacher live RGB development runs sealed.", flush=True)


if __name__ == "__main__":
    main()
