"""Explicit, non-resumable host entry point for the known-root AP development run.

Nothing happens on import. --check reads assets only; --execute is an explicit
physical/CUDA operation and must not be used without separate authorization.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/pc_ap_official_kernel_physical_dev_20261102_v1/report.json"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
OUTPUT = ROOT / "results/pc_ap_durable_host_20261102_codexexclusive_RUN_01"
REPORT_SHA = "f2c95637b9dda338bf7ed4165934ac384c4c78308cc0aa932ade776fd04acd91"
SOURCE_SHA = {
    "pc_ap_durable_development_runner_20261102_codexexclusive.py": "6519ee5ebc5fa0837dd74f88c737f59b28f9967c5652fab8875b5629c2477149",
    "pc_ap_durable_context_adapter_20261102_codexexclusive.py": "2bb9afc25faa23ab8753b1d93a32eeeca5074c28d0db772a9a994ed9a8059b27",
    "pc_ap_feedback_step_bridge_20261102_codexexclusive.py": "859ca39e9e6ec6de1b05491f1ae7dc48a3ad449ebd8a0cfb47cc4a63811aee02",
    "pc_ap_official_cem_kernel.py": "6ddfa22a333a5f899794085dc30604157541dcb9a8d995c33e5e3110fbb1f129",
    "pc_ap_reacher_cuda_adapter.py": "5bfbb539b6bca8511b16cb38c4affffb278326f48b05d7110ae337c19e10be23",
    "preflight_reacher_d2p_p1_reconfiguration.py": "10f54289b375b241a0546cb7b69b97e770c5fb45c996ed83612cb004e12e16fd",
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected):
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"SHA mismatch: {path}")
    return actual


def check_assets(policy, stats, dataset, *, report=REPORT, source_dir=None):
    """Read-only authentication of used bytes; does not load a model."""
    import h5py
    import hdf5plugin  # noqa: F401 - register the source HDF5 filters
    import numpy as np

    source_dir = source_dir or Path(__file__).parent
    require_hash(report, REPORT_SHA)
    reference = json.loads(Path(report).read_text())
    for name, digest in SOURCE_SHA.items():
        require_hash(source_dir / name, digest)
    require_hash(Path(policy) / "weights.pt", reference["weights_sha256"])
    require_hash(stats, reference["stats_sha256"])
    receipt = json.loads(Path(stats).read_text())
    if Path(receipt["source"]["dataset"]).resolve() != Path(dataset).resolve():
        raise ValueError("memory dataset differs from action-stat source")
    episodes = {}
    memory_info = {}
    with h5py.File(dataset, "r", swmr=True) as data:
        for episode in (0, 2, 3):
            offset, length = int(data["ep_offset"][episode]), int(data["ep_len"][episode])
            if offset < 0 or length < 31 or not np.all(data["ep_idx"][offset:offset + 31] == episode):
                raise ValueError("invalid memory episode provenance")
            frames = np.asarray(data["pixels"][offset:offset + 31])
            if len(frames) != 31 or frames.dtype != np.uint8:
                raise ValueError("invalid memory pixels")
            digest = hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()
            if digest != reference["memory_rgb_sha256"][str(episode)]:
                raise ValueError("memory pixels SHA mismatch")
            episodes[str(episode)] = frames
            memory_info[str(episode)] = {"offset": offset, "episode_length": length, "rgb_sha256": digest}
    return reference, episodes, memory_info


def _roots(reference, seed):
    rows = [x for x in reference["rows"] if x.get("seed") == seed]
    pairs = {(x.get("root_rgb_sha256"), x.get("goal_rgb_sha256"))
             for x in rows if x.get("status") == "complete"}
    if len(rows) != 2 or len(pairs) != 1 or len([x for x in rows if x.get("status") == "complete"]) != 2:
        raise ValueError("reference pair incomplete or inconsistent")
    return next(iter(pairs))


@contextlib.contextmanager
def predictor_meter(module, journal):
    """Persist real predictor attempts and counters, including raised calls."""
    original = module.ReacherCudaPredictor

    class MeteredPredictor(original):
        def __call__(self, actions):
            journal.emit({"kind": "predictor_call_intent", "requested_blocks": len(actions),
                          "blocks_before": self.predicted_blocks, "batches_before": self.prediction_batches})
            try:
                value = super().__call__(actions)
            except BaseException as exc:
                journal.emit({"kind": "predictor_call_failed", "blocks": self.predicted_blocks,
                              "batches": self.prediction_batches, "error_type": type(exc).__name__})
                raise
            journal.emit({"kind": "predictor_call_returned", "blocks": self.predicted_blocks,
                          "batches": self.prediction_batches})
            return value

    module.ReacherCudaPredictor = MeteredPredictor
    try:
        yield
    finally:
        module.ReacherCudaPredictor = original


def execute(policy, stats, dataset):
    # Validate before acquiring the advisory lock; revalidate inside it.
    check_assets(policy, stats, dataset)
    from .e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
    with gpu_lock(LOCK):
        reference, episodes, info = check_assets(policy, stats, dataset)
        OUTPUT.mkdir(exist_ok=False)
        from . import pc_ap_durable_context_adapter_20261102_codexexclusive as durable
        journal = durable.Journal(OUTPUT / "host.jsonl")
        try:
            journal.emit({"kind": "host_started", "reference_sha256": REPORT_SHA,
                          "memory": info, "policy": str(Path(policy).resolve()),
                          "stats": str(Path(stats).resolve()), "dataset": str(Path(dataset).resolve()),
                          "counts_note": "predictor attempts are recorded; other model operations are not fully metered"})
            import gymnasium as gym
            import numpy as np
            import torch
            from . import preflight_reacher_d2p_p1_reconfiguration as setup
            from . import pc_ap_durable_development_runner_20261102_codexexclusive as runner
            from . import pc_ap_reacher_cuda_adapter as predictor_module
            from .pc_ap_observation_memory import ObservationMemory
            from .run_pc_ap_reacher_cuda_smoke import encode_frames

            if not torch.cuda.is_available():
                raise RuntimeError("host CUDA required")
            setup.install_missing_field_compat()
            setup.validate_training_semantic_receipt(stats)
            mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
            mean, std = np.asarray(mean), np.asarray(std)
            if mean.shape != (10,) or std.shape != (10,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("invalid 5x2 action normalization")
            _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
            journal.emit({"kind": "normalization_validated", "mean": mean.tolist(), "std": std.tolist()})
            journal.emit({"kind": "model_load_intent"})
            model = setup.load_compatible_pretrained(policy, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            if before != reference["parameter_hash_before"]:
                raise ValueError("loaded parameter hash differs from reference")
            journal.emit({"kind": "model_loaded", "parameter_sha256": before})
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            encoded = {}
            for key, frames in episodes.items():
                journal.emit({"kind": "memory_encode_intent", "episode": key, "frames": len(frames)})
                encoded[key] = encode_frames(model, processor, frames)
                torch.cuda.synchronize()
                journal.emit({"kind": "memory_encode_returned", "episode": key, "frames": len(frames)})
            memory = ObservationMemory(encoded, ["live-dev-reset-20263160", "live-dev-reset-20263161"])
            for seed in runner.SEEDS:
                for arm in runner.ARMS:
                    arm_journal = durable.Journal(OUTPUT / f"{seed}_{arm}.jsonl")
                    env = None
                    try:
                        packets, history_index = setup.sample_history_packets(histories, seed=seed)
                        packets = np.asarray(packets, dtype=np.float32)
                        arm_journal.emit({"kind": "warmup_intent", "seed": seed, "arm": arm,
                                          "history_index": int(history_index), "shape": list(packets.shape),
                                          "actions_float32_hex": np.ascontiguousarray(packets).tobytes().hex()})
                        env = setup.ReacherGoalImageWrapper(
                            gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                            target_delta=(.65, -.65))
                        context = setup.collect_executed_context(
                            env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                            warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                        root = (context["root_rgb_sha256"], context["goal_rgb_sha256"])
                        arm_journal.emit({"kind": "warmup_returned", "root_rgb_sha256": root[0],
                                          "goal_rgb_sha256": root[1]})
                        if root != _roots(reference, seed):
                            raise ValueError("live root/goal differ from frozen reference")
                        def distance(e):
                            raw = e.unwrapped
                            return setup.p0._qpos_distance(np.asarray(raw.env.physics.data.qpos),
                                                           np.asarray(raw.env.task.target_qpos))
                        with predictor_meter(predictor_module, arm_journal):
                            planner = runner.cuda_planner(model, processor, memory, mean, std, env)
                            result = runner.run_arm(env, context, planner, arm_journal,
                                                    seed=seed, arm=arm, distance=distance)
                        if result["status"] != "complete":
                            raise RuntimeError(f"arm incomplete: {result}")
                        journal.emit({"kind": "arm_returned", "seed": seed, "arm": arm, "result": result})
                    except BaseException as exc:
                        arm_journal.emit({"kind": "host_arm_failed", "error_type": type(exc).__name__,
                                          "message": str(exc), "no_retry": True})
                        raise
                    finally:
                        if env is not None:
                            env.close()
                        arm_journal.close()
            if setup._parameter_hash(model) != before:
                raise RuntimeError("model parameters changed")
            check_assets(policy, stats, dataset)
            journal.emit({"kind": "host_completed", "parameter_sha256": before})
        except BaseException as exc:
            journal.emit({"kind": "host_failed", "error_type": type(exc).__name__,
                          "message": str(exc), "no_retry": True})
            raise
        finally:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="separately authorized physical and CUDA run")
    args = parser.parse_args()
    if args.execute:
        execute(args.policy, args.action_stats, args.dataset)
    else:
        _, _, info = check_assets(args.policy, args.action_stats, args.dataset)
        print(json.dumps({"status": "asset_check_only", "memory": info, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
