"""Fit and validate a frozen-latent Reacher qpos readout on official data.

The world model remains fully frozen.  This script fits only a closed-form
ridge observation calibration from RGB encoder latents to sin/cos joint
coordinates, using episode-disjoint official source frames.  The artifact is
not a planner or a dynamics repair: its sole purpose is to make the task
coordinate used for P1 explicit and independently falsifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability.reacher_d2p_packet_support import receipt_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_episode_rows(
    lengths: np.ndarray, offsets: np.ndarray, *, episodes: np.ndarray, frames_per_episode: int,
) -> np.ndarray:
    if frames_per_episode < 2 or episodes.ndim != 1 or not len(episodes):
        raise ValueError("qpos readout requires at least one episode and two frames per episode")
    rows = []
    for episode in episodes.tolist():
        length, offset = int(lengths[episode]), int(offsets[episode])
        if length < frames_per_episode:
            raise ValueError("selected Reacher episode is shorter than readout sample count")
        rows.append(offset + np.linspace(0, length - 1, frames_per_episode, dtype=np.int64))
    return np.concatenate(rows)


def _encode_rgb(
    model: torch.nn.Module, preprocessor: _Preprocessor, images: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    values = np.asarray(images, dtype=np.uint8)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Reacher qpos readout expects [row, height, width, RGB] source frames")
    latents = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = np.ascontiguousarray(values[start:start + batch_size])
            prepared = preprocessor._prepare_info({"pixels": batch[:, None]})
            encoded = model.encode({"pixels": prepared["pixels"].to(device)})["emb"][:, 0]
            latents.append(encoded.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(latents, axis=0)


def _angle_targets(qpos: np.ndarray) -> np.ndarray:
    angles = np.asarray(qpos, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 2 or not np.isfinite(angles).all():
        raise ValueError("Reacher qpos targets must be finite [row, two_joint] angles")
    return np.concatenate((np.sin(angles), np.cos(angles)), axis=1)


def fit_ridge_angle_readout(latents: np.ndarray, qpos: np.ndarray, *, alpha: float) -> dict[str, np.ndarray]:
    """Fit a standardized ridge map to sin/cos angles without a learned module."""

    x = np.asarray(latents, dtype=np.float64)
    y = _angle_targets(qpos)
    if x.ndim != 2 or len(x) != len(y) or len(x) < x.shape[1] + 2 or alpha < 0.0:
        raise ValueError("invalid frozen-latent ridge-readout calibration data")
    mean = x.mean(axis=0)
    scale = x.std(axis=0).clip(1.0e-6)
    normalized = (x - mean) / scale
    design = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"feature_mean": mean, "feature_scale": scale, "coefficient": coefficient}


def decode_qpos(artifact: dict[str, np.ndarray], latents: np.ndarray) -> np.ndarray:
    x = np.asarray(latents, dtype=np.float64)
    normalized = (x - artifact["feature_mean"]) / artifact["feature_scale"]
    pred = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1) @ artifact["coefficient"]
    return np.arctan2(pred[:, :2], pred[:, 2:])


def wrapped_delta(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (np.asarray(predicted) - np.asarray(target) + np.pi) % (2.0 * np.pi) - np.pi


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = wrapped_delta(predicted, target)
    return {
        "mean_joint_wrapped_abs_error": float(np.abs(delta).mean()),
        "max_joint_wrapped_abs_error": float(np.abs(delta).max()),
        "qpos_distance_rmse": float(np.sqrt(np.mean(np.square(delta).sum(axis=1)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026102550)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--frames-per-episode", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reacher qpos-readout calibration requires host CUDA exactly; CPU fallback is forbidden")
    if min(args.train_episodes, args.validation_episodes, args.frames_per_episode, args.batch_size) < 1 or args.ridge_alpha < 0.0:
        raise ValueError("Reacher qpos-readout sizes must be positive and ridge alpha nonnegative")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("qpos-readout output directory must be new")
    policy, adapter = args.policy.resolve(), args.adapter_artifact.resolve()
    dataset, dataset_sha256 = receipt_source(adapter)
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False); model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    with h5py.File(dataset, "r", swmr=True) as handle:
        lengths, offsets = np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"])
        if len(lengths) < args.train_episodes + args.validation_episodes:
            raise ValueError("official Reacher source has too few episodes for a disjoint calibration split")
        selected = np.random.default_rng(args.seed).choice(len(lengths), size=args.train_episodes + args.validation_episodes, replace=False)
        train_episodes = np.sort(selected[:args.train_episodes])
        validation_episodes = np.sort(selected[args.train_episodes:])
        train_rows = _sample_episode_rows(lengths, offsets, episodes=train_episodes, frames_per_episode=args.frames_per_episode)
        validation_rows = _sample_episode_rows(lengths, offsets, episodes=validation_episodes, frames_per_episode=args.frames_per_episode)
        train_latents = _encode_rgb(model, preprocessor, np.asarray(handle["pixels"][train_rows]), device, args.batch_size)
        validation_latents = _encode_rgb(model, preprocessor, np.asarray(handle["pixels"][validation_rows]), device, args.batch_size)
        train_qpos = np.asarray(handle["qpos"][train_rows], dtype=np.float64)
        validation_qpos = np.asarray(handle["qpos"][validation_rows], dtype=np.float64)
    artifact = fit_ridge_angle_readout(train_latents, train_qpos, alpha=args.ridge_alpha)
    validation_metrics = _metrics(decode_qpos(artifact, validation_latents), validation_qpos)
    # Declared before live deployment: both joint-coordinate accuracy and the
    # two-joint task distance must be small on source-held-out episodes.
    gate = bool(validation_metrics["mean_joint_wrapped_abs_error"] <= 0.25 and validation_metrics["qpos_distance_rmse"] <= 0.40)
    output.mkdir(parents=True)
    np.savez_compressed(output / "qpos_readout.npz", **artifact, task=np.asarray(["reacher_absolute_qpos_sincos_ridge_v1"]), dataset_sha256=np.asarray([dataset_sha256]))
    report = {
        "schema": "reacher_d2p_frozen_latent_qpos_readout.v1",
        "status": "completed",
        "scope": "episode-disjoint official-source calibration of a frozen encoder observation coordinate; no world-model parameter is updated",
        "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter_artifact": str(adapter), "adapter_sha256": _sha256(adapter), "official_dataset": str(dataset), "official_dataset_sha256": dataset_sha256},
        "protocol": {"device": str(device), "seed": args.seed, "train_episode_count": int(len(train_episodes)), "validation_episode_count": int(len(validation_episodes)), "frames_per_episode": args.frames_per_episode, "ridge_alpha": args.ridge_alpha, "target": "[sin(qpos_0), sin(qpos_1), cos(qpos_0), cos(qpos_1)]", "episode_disjoint": True, "declared_validation_gate": {"mean_joint_wrapped_abs_error_max": 0.25, "qpos_distance_rmse_max": 0.40}},
        "split": {"train_episode_ids_sha256": hashlib.sha256(train_episodes.tobytes()).hexdigest(), "validation_episode_ids_sha256": hashlib.sha256(validation_episodes.tobytes()).hexdigest(), "train_row_count": int(len(train_rows)), "validation_row_count": int(len(validation_rows))},
        "validation": {**validation_metrics, "gate_passed": gate},
        "artifact": {"path": str(output / "qpos_readout.npz"), "sha256": _sha256(output / "qpos_readout.npz")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
