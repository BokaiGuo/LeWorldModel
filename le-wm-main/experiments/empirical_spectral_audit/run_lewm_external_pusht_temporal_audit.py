"""Frozen LeWM temporal audit on the public PushT validation trajectories.

This evaluator uses only the released validation videos and action/state files
from the public JEPA-WM PushT archive.  It runs the actual frozen
``pusht/lewm`` checkpoint on image windows, measures autoregressive latent
prediction error at multiple horizons, and compares observed actions with a
zero-action negative control.  The public release's documented action
normalization (divide by 100, then use its published mean/std) is recorded
explicitly because the original LeWM training statistics are not present.

The result is an external frozen temporal diagnostic.  It is not retraining,
online environment control, or a claim that the public archive exactly
matches LeWM's original training distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUBLIC_ACTION_MEAN = torch.tensor([-0.0087, 0.0068], dtype=torch.float32)
PUBLIC_ACTION_STD = torch.tensor([0.2019, 0.2002], dtype=torch.float32)
FRAME_STRIDE = 5
CONTEXT_LENGTH = 3


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_starts(
    sequence_length: int,
    context_length: int = CONTEXT_LENGTH,
    max_horizon: int = 8,
    windows_per_episode: int = 2,
) -> list[int]:
    if sequence_length <= 0 or context_length <= 0 or max_horizon <= 0:
        raise ValueError("sequence and horizon parameters must be positive")
    obs_count = (int(sequence_length) - 1) // FRAME_STRIDE + 1
    required = int(context_length) + int(max_horizon)
    max_start = obs_count - required
    if max_start < 0:
        return []
    count = max(1, min(int(windows_per_episode), max_start + 1))
    return sorted({int(v) for v in np.linspace(0, max_start, count)})


def _standardize_public_actions(
    raw: torch.Tensor,
    normalization_mode: str = "public",
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
) -> torch.Tensor:
    if raw.ndim != 2 or raw.shape[-1] != 2:
        raise ValueError("raw actions must have shape (T, 2)")
    scaled = raw.float() / 100.0
    if normalization_mode == "public":
        mean, std = PUBLIC_ACTION_MEAN, PUBLIC_ACTION_STD
    elif normalization_mode == "val_zscore":
        if action_mean is None or action_std is None:
            raise ValueError("val_zscore requires action_mean and action_std")
        mean, std = action_mean, action_std
    elif normalization_mode == "raw_div100":
        mean, std = torch.zeros(2), torch.ones(2)
    else:
        raise ValueError(f"unsupported normalization mode: {normalization_mode}")
    return (scaled - mean) / std


def _make_action_blocks(
    raw_actions: torch.Tensor,
    start_obs: int,
    count: int,
    mode: str,
    normalization_mode: str = "public",
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = _standardize_public_actions(
        raw_actions,
        normalization_mode=normalization_mode,
        action_mean=action_mean,
        action_std=action_std,
    )
    if mode == "zero":
        normalized = torch.zeros_like(normalized)
    elif mode != "observed":
        raise ValueError(f"unsupported action mode: {mode}")
    blocks = []
    for obs_index in range(int(start_obs), int(start_obs) + int(count)):
        begin = obs_index * FRAME_STRIDE
        end = begin + FRAME_STRIDE
        block = normalized[begin:end]
        if block.shape[0] < FRAME_STRIDE:
            block = torch.zeros((FRAME_STRIDE, normalized.shape[-1]), dtype=normalized.dtype)
        blocks.append(block.reshape(-1))
    return torch.stack(blocks)


def _load_video_frames(path: Path, frame_indices: list[int]) -> torch.Tensor:
    from decord import VideoReader

    reader = VideoReader(str(path), num_threads=1)
    if not frame_indices or max(frame_indices) >= len(reader):
        raise ValueError(f"requested frames exceed video length for {path}")
    frames = torch.from_numpy(reader.get_batch(frame_indices).asnumpy()).float() / 255.0
    return rearrange(frames, "t h w c -> t c h w")


def _predict_horizons(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    horizons: list[int],
    context_length: int,
) -> dict[int, torch.Tensor]:
    max_horizon = max(horizons)
    context_emb = embedding[:, :context_length].clone()
    context_act = action_embedding[:, :context_length].clone()
    predictions: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for step in range(1, max_horizon + 1):
            pred = model.predict(context_emb, context_act)[:, -1]
            if step in horizons:
                predictions[step] = pred.detach().cpu()
            if step < max_horizon:
                next_action = action_embedding[:, context_length - 1 + step : context_length + step]
                context_emb = torch.cat((context_emb, pred[:, None]), dim=1)[:, -context_length:]
                context_act = torch.cat((context_act, next_action), dim=1)[:, -context_length:]
    return predictions


def _relative_commutator(target: np.ndarray, prediction: np.ndarray, ridge: float = 1e-3) -> float:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[0] < 3:
        return float("nan")
    x = target - target.mean(axis=0, keepdims=True)
    y = prediction - prediction.mean(axis=0, keepdims=True)
    covariance = x.T @ x / max(target.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / target.shape[1], 1e-6)
    operator = np.linalg.solve(covariance + ridge * scale * np.eye(target.shape[1]), x.T @ y).T
    commutator = covariance @ operator - operator @ covariance
    denominator = np.linalg.norm(covariance, ord="fro") * np.linalg.norm(operator, ord="fro")
    return float(np.linalg.norm(commutator, ord="fro") / max(float(denominator), 1e-12))


def _bootstrap_mean(values: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires at least two values")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(int(repeats), values.size))].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(seed),
    }


def _bootstrap_clustered(
    values: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap episode means so multiple windows from one episode stay paired."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("values and groups must be aligned one-dimensional arrays")
    unique_groups = np.asarray(sorted(set(groups.tolist())))
    if unique_groups.size < 2:
        raise ValueError("cluster bootstrap requires at least two groups")
    cluster_means = np.asarray(
        [values[groups == group].mean() for group in unique_groups], dtype=np.float64
    )
    report = _bootstrap_mean(cluster_means, repeats=repeats, seed=seed)
    report["bootstrap_unit"] = "episode"
    report["clusters"] = int(unique_groups.size)
    report["windows"] = int(values.size)
    return report


def _episode_records(
    model: torch.nn.Module,
    val_root: Path,
    episode_ids: list[int],
    horizons: list[int],
    windows_per_episode: int,
    device: torch.device,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(v) for v in pickle.load(handle)]
    from utils import get_img_preprocessor

    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    records: list[dict[str, Any]] = []
    vector_store: dict[str, list[np.ndarray]] = {}
    max_horizon = max(horizons)
    for episode_id in episode_ids:
        starts = _window_starts(
            lengths[episode_id],
            context_length=CONTEXT_LENGTH,
            max_horizon=max_horizon,
            windows_per_episode=windows_per_episode,
        )
        print(
            f"episode={episode_id} length={lengths[episode_id]} windows={len(starts)}",
            flush=True,
        )
        for start in starts:
            count = CONTEXT_LENGTH + max_horizon
            observation_indices = [FRAME_STRIDE * (start + i) for i in range(count)]
            raw_frames = _load_video_frames(
                val_root / "obses" / f"episode_{episode_id:03d}.mp4", observation_indices
            )
            pixels = preprocess({"pixels": raw_frames})["pixels"].unsqueeze(0).to(device)
            for mode in ("observed", "zero"):
                action_blocks = _make_action_blocks(
                    actions[episode_id],
                    start,
                    count,
                    mode,
                    normalization_mode=normalization_mode,
                    action_mean=action_mean,
                    action_std=action_std,
                )
                action_blocks = action_blocks.unsqueeze(0).to(device)
                with torch.inference_mode():
                    encoded = model.encode({"pixels": pixels, "action": action_blocks})
                    predictions = _predict_horizons(
                        model,
                        encoded["emb"],
                        encoded["act_emb"],
                        horizons,
                        CONTEXT_LENGTH,
                    )
                for horizon in horizons:
                    target = encoded["emb"][:, CONTEXT_LENGTH - 1 + horizon].detach().cpu()
                    prediction = predictions[horizon]
                    error = float(torch.square(prediction - target).mean())
                    key = f"{mode}/h{horizon}"
                    vector_store.setdefault(f"{key}/target", []).append(target.numpy()[0])
                    vector_store.setdefault(f"{key}/prediction", []).append(prediction.numpy()[0])
                    records.append(
                        {
                            "episode_id": int(episode_id),
                            "start_observation": int(start),
                            "mode": mode,
                            "horizon": int(horizon),
                            "mse": error,
                        }
                    )
    arrays = {key: np.stack(values) for key, values in vector_store.items()}
    return records, arrays


def _summarize(records: list[dict[str, Any]], vectors: dict[str, np.ndarray], horizons: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_mode_horizon": {}, "paired_action_control": {}}
    for mode in ("observed", "zero"):
        for horizon in horizons:
            values = np.asarray(
                [row["mse"] for row in records if row["mode"] == mode and row["horizon"] == horizon],
                dtype=np.float64,
            )
            groups = np.asarray(
                [row["episode_id"] for row in records if row["mode"] == mode and row["horizon"] == horizon]
            )
            target = vectors[f"{mode}/h{horizon}/target"]
            prediction = vectors[f"{mode}/h{horizon}/prediction"]
            summary["by_mode_horizon"][f"{mode}/h{horizon}"] = {
                "mse": _bootstrap_clustered(values, groups, repeats=2000, seed=20261003 + horizon),
                "windows": int(values.size),
                "relative_commutator": _relative_commutator(target, prediction),
            }
    for horizon in horizons:
        observed = np.asarray(
            [row["mse"] for row in records if row["mode"] == "observed" and row["horizon"] == horizon],
            dtype=np.float64,
        )
        zero = np.asarray(
            [row["mse"] for row in records if row["mode"] == "zero" and row["horizon"] == horizon],
            dtype=np.float64,
        )
        groups = np.asarray(
            [row["episode_id"] for row in records if row["mode"] == "observed" and row["horizon"] == horizon]
        )
        if observed.shape != zero.shape:
            raise ValueError("observed and zero-action rows are not paired")
        summary["paired_action_control"][f"h{horizon}"] = {
            "zero_minus_observed_mse": _bootstrap_clustered(
                zero - observed, groups, repeats=2000, seed=20262003 + horizon
            ),
            "observed_mse_mean": float(observed.mean()),
            "zero_mse_mean": float(zero.mean()),
            "fraction_zero_worse": float(np.mean(zero > observed)),
        }
    return summary


def run_report(
    val_root: Path,
    archive: Path,
    policy: str,
    output: Path,
    horizons: list[int],
    windows_per_episode: int,
    device_name: str,
    normalization_mode: str,
) -> dict[str, Any]:
    if not (val_root / "states.pth").exists() or not (val_root / "rel_actions.pth").exists():
        raise FileNotFoundError("public validation root lacks states.pth or rel_actions.pth")
    if sorted(set(horizons)) != list(horizons) or min(horizons) < 1:
        raise ValueError("horizons must be sorted positive integers")
    device = torch.device(device_name)
    from experiments.io_utils import load_policy

    model = load_policy(policy, device=str(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    all_actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float() / 100.0
    if normalization_mode == "public":
        action_mean, action_std = PUBLIC_ACTION_MEAN, PUBLIC_ACTION_STD
    elif normalization_mode == "val_zscore":
        action_mean = all_actions.mean(dim=(0, 1))
        action_std = all_actions.std(dim=(0, 1)).clamp_min(1e-6)
    elif normalization_mode == "raw_div100":
        action_mean, action_std = torch.zeros(2), torch.ones(2)
    else:
        raise ValueError(f"unsupported normalization mode: {normalization_mode}")
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(v) for v in pickle.load(handle)]
    episode_ids = list(range(len(lengths)))
    records, vectors = _episode_records(
        model,
        val_root,
        episode_ids,
        horizons,
        windows_per_episode,
        device,
        normalization_mode,
        action_mean,
        action_std,
    )
    summary = _summarize(records, vectors, horizons)
    vectors_path = output.with_name("external_temporal_vectors.npz")
    np.savez_compressed(vectors_path, **vectors)
    report = {
        "artifact": "lewm_external_public_pusht_temporal_audit",
        "status": "frozen_external_temporal_diagnostic",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "split": "public validation directory",
            "episodes": int(len(episode_ids)),
            "episode_ids": episode_ids,
            "sequence_lengths": lengths,
        },
        "model": {
            "policy": policy,
            "device": str(device),
            "checkpoint": "/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt",
            "checkpoint_sha256": _sha256(Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt")),
        },
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": horizons,
            "windows_per_episode": int(windows_per_episode),
            "action_modes": ["observed", "zero"],
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": (
                    "public JEPA-WM PushT dataset loader constants"
                    if normalization_mode == "public"
                    else "validation split statistics"
                    if normalization_mode == "val_zscore"
                    else "unit scale after raw action division"
                ),
                "original_lewm_training_statistics_available": False,
            },
            "target": "encoded future LeWM embedding",
            "metric": "mean squared error between autoregressive predicted and encoded future embeddings",
            "vectors": str(vectors_path),
        },
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "This is an external frozen temporal diagnostic of the released pusht/lewm checkpoint.",
            "The public archive is an independent validation split, but its exact action normalization and data-generation lineage relative to LeWM training are not fully recoverable.",
            "The zero-action comparison is an input ablation, not a causal environment intervention.",
            "No online environment rollout, planning success, retraining gain, or universal world-model ranking is claimed.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Frozen LeWM on external public PushT validation videos",
        "",
        "The released `pusht/lewm` checkpoint is evaluated on the public validation videos with observed actions and a zero-action input ablation. The target is the encoded future embedding from the same frozen LeWM encoder.",
        "",
        "| mode | horizon | windows | MSE mean | 95% bootstrap CI | relative commutator |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for key, value in report["summary"]["by_mode_horizon"].items():
        mode, horizon = key.split("/h")
        mse = value["mse"]
        lines.append(
            f"| {mode} | {horizon} | {value['windows']} | {mse['mean']:.6g} | "
            f"[{mse['ci95'][0]:.6g}, {mse['ci95'][1]:.6g}] | {value['relative_commutator']:.6g} |"
        )
    lines.extend(["", "## Paired zero-action control", "", "| horizon | observed MSE | zero MSE | zero minus observed | 95% CI | fraction zero worse |", "| ---: | ---: | ---: | ---: | --- | ---: |"])
    for key, value in report["summary"]["paired_action_control"].items():
        delta = value["zero_minus_observed_mse"]
        lines.append(
            f"| {key[1:]} | {value['observed_mse_mean']:.6g} | {value['zero_mse_mean']:.6g} | "
            f"{delta['mean']:.6g} | [{delta['ci95'][0]:.6g}, {delta['ci95'][1]:.6g}] | {value['fraction_zero_worse']:.3f} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument(
        "--action-normalization",
        choices=["public", "val_zscore", "raw_div100"],
        default="public",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.time()
    report = run_report(
        val_root=args.val_root,
        archive=args.archive,
        policy=args.policy,
        output=args.output,
        horizons=args.horizons,
        windows_per_episode=args.windows_per_episode,
        device_name=args.device,
        normalization_mode=args.action_normalization,
    )
    report["protocol_elapsed_seconds"] = float(time.time() - started)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
