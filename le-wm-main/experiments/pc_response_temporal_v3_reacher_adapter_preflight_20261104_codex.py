"""Preflight the Reacher v3 frozen-model action/history/latent adapter."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import pc_response_temporal_v3_reacher_adapter as adapter
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import pc_response_temporal_feedback_v3 as method

ROOT = host.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_REACHER_ADAPTER_PREFLIGHT_PROTOCOL_20261104_CODEX.md"
SOURCE = ROOT / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json"
PREFIX = ROOT / "results/pc_response_temporal_v3_reacher_adapter_preflight_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
CONTEXT = PREFIX.with_name(PREFIX.name + "_context.npz")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")


def dependencies(source):
    paths = (Path(__file__), PROTOCOL, SOURCE, Path(adapter.__file__), Path(method.__file__),
             Path(setup.__file__), Path(p0.__file__), Path(host.__file__),
             Path(source["weights_path"]), Path(source["action_stats_path"]))
    return {str(path.resolve()): host.bridge.sha_file(path) for path in paths}


def main():
    if any(path.exists() for path in (MANIFEST, CONTEXT, RESULT)):
        raise FileExistsError("existing or partial preflight")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    source = json.loads(SOURCE.read_text())
    hashes = dependencies(source)
    if (hashes[str(Path(source["weights_path"]).resolve())] != source["weights_sha256"]
            or hashes[str(Path(source["action_stats_path"]).resolve())] != source["action_stats_sha256"]):
        raise ValueError("checkpoint or action statistics mismatch")
    host.write_new(MANIFEST, {"dependencies": hashes, "method": method.METHOD, "root_seed": 20263160,
                              "scope": "frozen model adapter preflight; no policy execution",
                              "source_closure_complete": False})
    print("Reacher v3 adapter preflight on one exposed root; no policy actions.", flush=True)
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
        parameter_hash = setup._parameter_hash(model)
        if parameter_hash != source["model_parameter_hash_expected"]:
            raise ValueError("frozen model parameter mismatch")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                            target_delta=(.65, -.65))
        try:
            if (not np.array_equal(env.action_space.low, np.full(2, -1))
                    or not np.array_equal(env.action_space.high, np.full(2, 1))):
                raise ValueError("raw action domain differs from v3 [-1,1]")
            p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
            packets, index = setup.sample_history_packets(histories, seed=20263160)
            context = setup.collect_executed_context(env, seed=20263160, history_len=3, action_block=5,
                warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            expected = next(row for row in source["roots"] if row["seed"] == 20263160)
            if (int(index) != expected["history_index"] or context["root_rgb_sha256"] != expected["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != expected["goal_rgb_sha256"]):
                raise ValueError("root, goal or history differs from sealed source")
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
                                                 np.asarray(context["history_action_blocks"], np.float32), device)
            with torch.no_grad():
                encoded, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                chart = adapter.goal_chart(encoded, goal_latent, 2027111800)
                np.testing.assert_allclose((chart.T @ chart).cpu().numpy(), np.eye(2), rtol=0, atol=1e-5)
                target = (goal_latent @ chart).cpu().numpy()
                rows, rng = [], np.random.default_rng(2027111801)
                for horizon in (5, 10, 15):
                    actions = rng.uniform(-.9, .9, size=(3, horizon, 2)).astype(np.float32)
                    actions[0] = 0
                    future = adapter.raw_to_packets(actions, mean, std, device=device)
                    restored = (future.cpu().numpy() * np.asarray(std) + np.asarray(mean)).reshape(actions.shape)
                    np.testing.assert_allclose(restored, actions, rtol=0, atol=2e-7)
                    reference = p0._terminal_batch(model, pixels, goal, history, future,
                                                    action_lower=lower, action_upper=upper)
                    cached = adapter.cached_terminal(model, encoded, history, future, lower, upper)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(cached, reference, rtol=2e-5, atol=2e-5)
                    projected = cached @ chart
                    scores = (cached - goal_latent).square().mean(dim=1)
                    if not torch.isfinite(projected).all() or not torch.isfinite(scores).all():
                        raise ValueError("nonfinite prediction")
                    rows.append({"horizon": horizon, "raw": actions.tolist(), "packets": future.cpu().tolist(),
                                 "reference": reference.cpu().tolist(), "cached": cached.cpu().tolist(),
                                 "max_abs_difference": float(torch.max(torch.abs(cached-reference))),
                                 "positions": projected.cpu().tolist(), "scores": scores.cpu().tolist()})
            with CONTEXT.open("xb") as stream:
                np.savez_compressed(stream, frames=np.stack(context["frames"]), root=context["root_frame"],
                                    goal=context["goal"], history=np.asarray(context["history_action_blocks"]),
                                    warmup=context["warmup_raw_steps"])
            if setup._parameter_hash(model) != parameter_hash or dependencies(source) != hashes:
                raise ValueError("model or dependency drift")
            host.write_new(RESULT, {"manifest_sha256": host.bridge.sha_file(MANIFEST),
                "context_sha256": host.bridge.sha_file(CONTEXT), "model_parameter_hash": parameter_hash,
                "root_sha256": context["root_rgb_sha256"], "goal_sha256": context["goal_rgb_sha256"],
                "history_index": int(index), "history_rgb_sha256": context["history_rgb_sha256"],
                "warmup_sha256": context["warmup_actions_sha256"], "mean": mean.tolist(), "std": std.tolist(),
                "chart": chart.cpu().tolist(), "target": target.tolist(), "rows": rows,
                "cuda_allocation": allocation, "source_closure_complete": False,
                "scope": "preflight only; no new policy actions or scientific gate"})
        finally:
            env.close()
    print("Reacher adapter preflight sealed.", flush=True)


if __name__ == "__main__":
    main()
