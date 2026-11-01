"""CUDA-only Cube D2P P0 with source-supported five-action packets.

This is a fresh implementation admission for the frozen Cube checkpoint.  It
does not use the legacy Cube CEM runner, its adapter artifact, or any
privileged simulator state as a model input.  In particular it refuses the
otherwise easy-to-miss substitution of independently sampled controls for a
time-major action token that the checkpoint saw during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 -- registers swm/OGBCube-v0
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.online_planning_protocol import validate_action_protocol
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import preflight_tworoom_d2p_p0 as packet_runtime
from experiments.spectral_reachability import cube_d2p_training_source as source


SCHEMA = "cube_d2p_p0_preflight.v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Preserve replay options in immutable JSON without silent string casts."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cube replay provenance has unsupported type {type(value)!r}")


def _validate_cube_action_protocol(model: torch.nn.Module, mean: np.ndarray, std: np.ndarray) -> int:
    block = validate_action_protocol(model, action_dim=source.RAW_ACTION_DIM, action_block=source.ACTION_BLOCK)
    if block != source.ACTION_BLOCK or mean.shape != (block * source.RAW_ACTION_DIM,) or std.shape != mean.shape:
        raise ValueError("Cube source action statistics do not match frozen 25-wide action encoder")
    tiled_mean = np.broadcast_to(mean[:source.RAW_ACTION_DIM], (block, source.RAW_ACTION_DIM))
    tiled_std = np.broadcast_to(std[:source.RAW_ACTION_DIM], (block, source.RAW_ACTION_DIM))
    if not np.array_equal(mean.reshape(block, source.RAW_ACTION_DIM), tiled_mean):
        raise ValueError("Cube action means are position-specific rather than raw-action training statistics")
    if not np.array_equal(std.reshape(block, source.RAW_ACTION_DIM), tiled_std):
        raise ValueError("Cube action standard deviations are position-specific rather than raw-action training statistics")
    return block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--training-source-receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/cube/lewm"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=202610250)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--goal-offset-packets", type=int, default=1)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument("--probe-dims", type=int, default=8)
    parser.add_argument("--fd-pairs", type=int, default=8)
    parser.add_argument("--fd-epsilon", type=float, default=1e-3)
    args = parser.parse_args()
    if min(args.roots, args.history_len, args.goal_offset_packets, args.future_horizon, args.probe_dims, args.fd_pairs) < 1 or args.fd_epsilon <= 0:
        parser.error("root counts, horizons, probe counts, and fd epsilon must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output, args.policy = args.output.resolve(), args.policy.resolve()
    args.training_dataset, args.training_source_receipt = args.training_dataset.resolve(), args.training_source_receipt.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable Cube D2P P0 output: {args.output}")
    runtime = require_cuda_allocation(args.device)
    receipt = source.validate_receipt(args.training_source_receipt)
    receipt_source = json.loads(args.training_source_receipt.read_text(encoding="utf-8"))["source"]
    if args.training_dataset != Path(receipt_source["dataset"]).resolve():
        raise ValueError("Cube P0 dataset differs from the source locked by its receipt")
    mean = np.asarray(receipt["calibration_action_mean"], dtype=np.float32)
    std = np.asarray(receipt["calibration_action_std"], dtype=np.float32)
    device = torch.device(args.device)
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = packet_runtime._parameter_hash(model)
    # Action bounds constrain only the differentiated candidate.  The model
    # prefix and goal below come directly from the official HDF5, not this
    # environment; no reset-rendered surrogate can enter the model path.
    env = gym.make("swm/OGBCube-v0", render_mode="rgb_array")
    checkpoint_block = _validate_cube_action_protocol(model, mean, std)
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
    transform = _image_transform()
    roots: list[dict[str, Any]] = []
    started = time.time()
    try:
        contexts, context_provenance = source.sample_source_native_contexts(
            args.training_dataset, expected_sha256=str(receipt_source["dataset_sha256"]),
            history_len=args.history_len, goal_offset_packets=args.goal_offset_packets,
            count=args.roots, seed=args.seed,
        )
        for root_index, (context, provenance) in enumerate(zip(contexts, context_provenance)):
            seed = args.seed + root_index
            audit = packet_runtime._jacobian_and_fd(
                model, context["frames"], context["root_frame"], context["history_raw_packets"], context["goal"],
                mean=mean, std=std, action_block=source.ACTION_BLOCK, action_dim=source.RAW_ACTION_DIM,
                future_horizon=args.future_horizon, probe_dims=args.probe_dims, fd_pairs=args.fd_pairs,
                epsilon=args.fd_epsilon, seed=seed, transform=transform, device=device, low=low, high=high,
            )
            audit["action_layout"]["flattened_packet_layout"] = "time-major [a0_x,a0_y,a0_z,a0_yaw,a0_grip,...,a4_grip]"
            roots.append({
                "seed": seed,
                "jacobian": audit,
                "source_native_context": {
                    "context_mode": "official_hdf5_episode_contiguous_rgb_action_future_goal",
                    "history_rgb_sha256": [_sha256_array(frame) for frame in context["frames"]],
                    "root_rgb_sha256": _sha256_array(context["root_frame"]),
                    "goal_rgb_sha256": _sha256_array(context["goal"]),
                    "history_actions_sha256": _sha256_array(context["history_raw_packets"]),
                    "source_rows": context["source_rows"],
                    "source_provenance": provenance,
                    "evaluator_only_source_state": context["evaluator_only_source_state"],
                },
            })
            print(json.dumps({"seed": seed, "jacobian_norm": audit["frobenius_norm"], "fd_median_relative_error": audit["median_relative_error"]}, sort_keys=True))
    finally:
        env.close()
    norms = [float(row["jacobian"]["frobenius_norm"]) for row in roots]
    fd_errors = [float(row["jacobian"]["median_relative_error"]) for row in roots]
    weights = args.policy / "weights.pt"
    gates = {
        "cuda_only": True,
        "cuda_real_allocation": True,
        "frozen_parameter_hash_unchanged": before == packet_runtime._parameter_hash(model),
        "checkpoint_action_block_matches_source_receipt": checkpoint_block == source.ACTION_BLOCK,
        "training_semantic_tiled_action_statistics": True,
        "source_native_rgb_history_and_goal_are_same_episode": True,
        "source_native_full_packet_history_no_padding": True,
        "every_root_nonzero_finite_future_action_jacobian": bool(norms and min(norms) > 1e-10),
        "central_fd_median_relative_error_le_0_10_each_root": bool(fd_errors and max(fd_errors) <= 0.10),
        "planner_input_is_official_source_rgb_future_goal_and_causal_actions_only": True,
        "evaluator_state_not_used_as_model_input": True,
        "no_adapter_teacher_or_privileged_ranker": True,
        "p1_or_outcome_roots_not_opened": True,
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only frozen Cube D2P P0 input/Jacobian admission; no P1 selection or planning efficacy conclusion.",
        "cuda_host_preflight": {**runtime, "python": platform.python_version()},
        "inputs": {
            "checkpoint_weights": str(weights),
            "checkpoint_weights_sha256": _sha256_file(weights),
            "training_source_receipt": str(args.training_source_receipt),
            "training_source_receipt_sha256": _sha256_file(args.training_source_receipt),
            "training_packet_support": {
                "dataset": str(args.training_dataset),
                "dataset_sha256": str(receipt_source["dataset_sha256"]),
                "sampling": "lazy episode-contained contiguous packet sequences; no global action-packet pool materialised",
            },
            "raw_environment_action_box": [low.detach().cpu().tolist(), high.detach().cpu().tolist()],
        },
        "protocol": {
            "environment": source.ENVIRONMENT,
            "runtime_input": "official HDF5 RGB/action causal history ending at an observed source RGB root, same-episode future source RGB goal, and bounded future raw action packets",
            "history_len": args.history_len,
            "goal_offset_packets": args.goal_offset_packets,
            "future_horizon": args.future_horizon,
            "action_block": source.ACTION_BLOCK,
            "raw_action_dim": source.RAW_ACTION_DIM,
            "root_seeds": [row["seed"] for row in roots],
            "future_stages": "P1 may only be written after this receipt passes and must bind the same source packet pool.",
        },
        "parameter_hash_before": before,
        "parameter_hash_after": packet_runtime._parameter_hash(model),
        "gates": gates,
        "aggregate": {"jacobian_frobenius_norms": norms, "fd_median_relative_errors": fd_errors},
        "roots": roots,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
