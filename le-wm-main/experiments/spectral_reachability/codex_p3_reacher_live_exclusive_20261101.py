"""Exclusive single-chunk Reacher wiring probe, not a P3 evaluation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import codex_p3_chunk_extension_20261101 as kernel
from experiments.spectral_reachability import codex_p3_live_chunk_adapter_20261101 as adapter


def write_once(path, value):
    data = json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n"
    with Path(path).open("x") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_actions(raw, mean, std, low, high, block):
    """Construct model tokens from the exact physical actions, time-major."""
    raw = np.asarray(raw, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    low, high = np.asarray(low), np.asarray(high)
    if (raw.ndim != 2 or type(block) is not int or block <= 0
            or raw.shape[0] == 0 or raw.shape[0] % block
            or mean.shape != (block * raw.shape[1],) or std.shape != mean.shape
            or not np.isfinite(raw).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or np.any(raw < low) or np.any(raw > high)):
        raise ValueError("invalid physical actions or normalization")
    tokens = (raw.reshape(-1, mean.size) - mean) / std
    if not np.isfinite(tokens).all():
        raise ValueError("nonfinite normalized action tokens")
    return np.ascontiguousarray(tokens), np.ascontiguousarray(raw)


def run_bound_chunk(*, output, model, env, states, history_actions, raw, mean, std,
                    low, high, goal, projection, block, config, arm, meter):
    tokens, raw = bind_actions(raw, mean, std, low, high, block)
    seal = {"schema": "p3_action_binding.v1", "arm": arm,
            "config": vars(config), "token_steps": block,
            "normalization_mean": np.asarray(mean).tolist(),
            "normalization_std": np.asarray(std).tolist(),
            "physical_actions": raw.tolist(), "normalized_tokens": tokens.tolist(),
            "rule": "float32 time-major (raw - mean) / std",
            "physical_bounds_low": np.asarray(low).tolist(),
            "physical_bounds_high": np.asarray(high).tolist()}
    seal_id = write_once(Path(output) / "action_binding.json", seal)
    # Reconstruct execution and model input from the exact sealed values.
    sealed = json.loads((Path(output) / "action_binding.json").read_text())
    verified_tokens, verified_raw = bind_actions(
        sealed["physical_actions"], sealed["normalization_mean"],
        sealed["normalization_std"], low, high, block)
    if not np.array_equal(verified_tokens, np.asarray(sealed["normalized_tokens"], dtype=np.float32)):
        raise ValueError("sealed token/physical action mismatch")
    token_tensor = torch.as_tensor(verified_tokens, device=states.device, dtype=states.dtype)[None]
    result = adapter.run_chunk(model=model, env=env, states=states,
        history_actions=history_actions, action_tokens=token_tensor,
        physical_actions=verified_raw, goal=goal, projection=projection,
        token_steps=block, config=config, arm=arm, meter=meter)
    result["action_binding_sha256"] = seal_id
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("fixed", "dynamic"), required=True)
    parser.add_argument("--min-improvement", type=float, required=True)
    parser.add_argument("--max-disagreement", type=float, required=True)
    args = parser.parse_args()
    config = kernel.Config(5, 5, 2, args.min_improvement, args.max_disagreement)
    args.output.mkdir(parents=True, exist_ok=False)
    policy_path = Path(args.policy).resolve()
    policy_files = sorted(p for p in policy_path.rglob("*") if p.is_file()) if policy_path.is_dir() else [policy_path]
    if not policy_files:
        raise ValueError("checkpoint contains no files")
    sources = [Path(__file__).resolve(), Path(adapter.__file__).resolve(),
               Path(kernel.__file__).resolve(), args.adapter_artifact.resolve(), *policy_files]
    exp = Path(__file__).resolve().parents[1]
    sources += [exp / name for name in (
        "spectral_reachability/run_reacher_d2p_p0_goal_progress.py",
        "spectral_reachability/reacher_d2p_context.py",
        "spectral_reachability/preflight_reacher_d2p_p0.py",
        "audit_live_reacher_shared_candidate_ranking.py", "checkpoint_compat.py",
        "run_live_reacher_planning.py", "run_live_cube_action_adapter_planning.py")]
    fingerprints = {str(path): file_hash(path) for path in sources}
    # Protocol exists before model inference, environment creation or warmup.
    write_once(args.output / "protocol.json", {
        "schema": "p3_reacher_live_exclusive.v1", "args": vars(args) | {
            "adapter_artifact": str(args.adapter_artifact.resolve()), "output": str(args.output.resolve())},
        "config": vars(config), "candidate": "15 zero physical actions",
        "warmup": "existing seeded full-packet Reacher context",
        "projection": "identity latent", "predict_cap": 8, "file_sha256": fingerprints,
        "scope": "single exploratory chunk; no fairness, efficacy or safety claim"})
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
    from experiments.gpu_preflight import require_cuda_allocation
    import gymnasium as gym

    counts = {name: {"attempted": 0, "returned": 0, "samples_attempted": 0,
                     "input_elements_attempted": 0}
              for name in ("encoder", "action_encoder")}
    step_counts = {"attempted": 0, "returned": 0}
    hooks, env, result, failure, root_identity = [], None, None, None, None
    meter = kernel.Meter({axis: 8 if axis in ("predict", "predictor_samples") else 0
                          for axis in kernel.AXES})
    started = time.perf_counter()
    try:
        require_cuda_allocation("cuda")
        torch.cuda.reset_peak_memory_stats()
        p0.install_missing_field_compat()
        p0.validate_training_semantic_receipt(args.adapter_artifact)
        mean, std = p0._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
        model = p0.load_compatible_pretrained(args.policy, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        for name in counts:
            module = getattr(model, name)
            def before(_module, inputs, name=name):
                counts[name]["attempted"] += 1
                counts[name]["samples_attempted"] += int(inputs[0].shape[0])
                counts[name]["input_elements_attempted"] += int(inputs[0].numel())
            def after(_module, _inputs, _output, name=name):
                counts[name]["returned"] += 1
            hooks.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
        class CountSteps(gym.Wrapper):
            def step(self, action):
                step_counts["attempted"] += 1
                value = self.env.step(action)
                step_counts["returned"] += 1
                return value
        env = CountSteps(p0.ReacherGoalImageWrapper(
            gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array")))
        action_dim = int(np.prod(env.action_space.shape))
        p0._validate_reacher_action_protocol(model, mean, std, action_dim=action_dim, action_block=5)
        context = p0.collect_executed_context(env, seed=args.seed, history_len=3, action_block=5)
        root_identity = {key: context[key] for key in
                         ("root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")}
        preprocessor = p0._Preprocessor(p0.ActionNormalizer(mean, std))
        pixels, goal_rgb, history = p0._prepare(preprocessor, context["frames"], context["root_frame"],
            context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
        states = p0._latent_context(model, pixels, history)
        goal = p0._goal_latent(model, goal_rgb)
        if any(file_hash(path) != digest for path, digest in fingerprints.items()):
            raise ValueError("presealed file changed")
        raw = np.zeros((15, action_dim), dtype=np.float32)
        result = run_bound_chunk(output=args.output, model=model, env=env, states=states,
            history_actions=history[:, 1:], raw=raw, mean=mean, std=std,
            low=env.action_space.low, high=env.action_space.high, goal=goal,
            projection=torch.eye(states.shape[-1], device=states.device), block=5,
            config=config, arm=args.arm, meter=meter)
        torch.cuda.synchronize()
        if any(file_hash(path) != digest for path, digest in fingerprints.items()):
            raise ValueError("presealed file changed during chunk")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "physical_counts": getattr(exc, "p3_physical_counts", None)}
    finally:
        for hook in hooks:
            hook.remove()
        if env is not None:
            env.close()
        write_once(args.output / "receipt.json", {
            "result": result, "failure": failure, "root_identity": root_identity,
            "elapsed_seconds": time.perf_counter() - started,
            "model_module_counts": counts, "all_env_steps_including_warmup": step_counts,
            "extension_meter": {"attempted": meter.attempted, "returned": meter.returned,
                                "events": meter.events, "caps": meter.caps},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            "resource_limitations": "No energy, FLOPs, CPU peak memory, model-internal operator, or asynchronous failure accounting; no total-compute equality claim.",
            "p3_admitted": False, "fairness_established": False, "success_claim": False})
    if failure:
        raise RuntimeError(f"probe failed; receipt retained: {failure}")


if __name__ == "__main__":
    main()
