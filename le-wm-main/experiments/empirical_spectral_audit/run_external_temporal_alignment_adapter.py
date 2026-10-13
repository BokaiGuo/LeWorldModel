"""Frozen external PushT temporal adapter with episode-level alignment controls.

The released LeWM checkpoint is evaluated on public validation videos.  A tiny
residual adapter is trained on one-step latent transitions from training
episodes and then evaluated autoregressively on held-out episodes.  The base
LeWM encoder and predictor stay frozen.  This closes the gap between the
row-level real-latent adapter (one-step only) and the synthetic multi-step
alignment experiments without claiming end-to-end retraining or online
control.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    _load_video_frames,
    _make_action_blocks,
    _window_starts,
)
from experiments.empirical_spectral_audit.run_real_latent_alignment_adapter import ResidualAdapter


@dataclass(frozen=True)
class TemporalAdapterConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    max_horizon: int = 8
    windows_per_episode: int = 2
    alignment_bins: int = 2
    alignment_min_samples: int = 8
    alignment_ridge: float = 1e-3
    split_seed: int = 3072


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return bool(math.isfinite(float(value)))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalization_parameters(
    actions: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor, str]:
    scaled = actions.float() / 100.0
    if mode == "public":
        return PUBLIC_ACTION_MEAN, PUBLIC_ACTION_STD, "public JEPA-WM loader constants"
    if mode == "val_zscore":
        return (
            scaled.mean(dim=(0, 1)),
            scaled.std(dim=(0, 1)).clamp_min(1e-6),
            "statistics computed from the external validation actions",
        )
    if mode == "raw_div100":
        return torch.zeros(2), torch.ones(2), "unit scale after raw action division"
    raise ValueError(f"unsupported action normalization mode: {mode}")


def _collect_windows(
    model: torch.nn.Module,
    val_root: Path,
    episode_ids: list[int],
    config: TemporalAdapterConfig,
    device: torch.device,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> list[dict[str, Any]]:
    """Encode true context/future windows once and retain only small tensors."""

    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(v) for v in pickle.load(handle)]
    from utils import get_img_preprocessor

    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    model.eval()
    windows: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        starts = _window_starts(
            lengths[episode_id],
            context_length=CONTEXT_LENGTH,
            max_horizon=config.max_horizon,
            windows_per_episode=config.windows_per_episode,
        )
        print(f"episode={episode_id} length={lengths[episode_id]} windows={len(starts)}", flush=True)
        for start in starts:
            count = CONTEXT_LENGTH + config.max_horizon
            frame_indices = [FRAME_STRIDE * (start + i) for i in range(count)]
            frames = _load_video_frames(
                val_root / "obses" / f"episode_{episode_id:03d}.mp4", frame_indices
            )
            pixels = preprocess({"pixels": frames})["pixels"].unsqueeze(0).to(device)
            blocks = _make_action_blocks(
                actions[episode_id],
                start,
                count,
                "observed",
                normalization_mode=normalization_mode,
                action_mean=action_mean,
                action_std=action_std,
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                encoded = model.encode({"pixels": pixels, "action": blocks})
                embeddings = encoded["emb"].detach().cpu()[0]
                action_embeddings = encoded["act_emb"].detach().cpu()[0]
                context = encoded["emb"][:, :CONTEXT_LENGTH]
                context_actions = encoded["act_emb"][:, :CONTEXT_LENGTH]
                base_next = model.predict(context, context_actions)[:, -1].detach().cpu()[0]
            source = embeddings[CONTEXT_LENGTH - 1]
            targets = embeddings[CONTEXT_LENGTH : CONTEXT_LENGTH + config.max_horizon]
            future_actions = action_embeddings[
                CONTEXT_LENGTH - 1 : CONTEXT_LENGTH - 1 + config.max_horizon
            ]
            windows.append(
                {
                    "episode_id": int(episode_id),
                    "start_observation": int(start),
                    "context": embeddings[:CONTEXT_LENGTH].float(),
                    "context_actions": action_embeddings[:CONTEXT_LENGTH].float(),
                    "source": source.float(),
                    "base_delta": (base_next - source).float(),
                    "target": targets[0].float(),
                    "targets": targets.float(),
                    "future_actions": future_actions.float(),
                }
            )
    if not windows:
        raise ValueError("no valid temporal windows were collected")
    return windows


def _split_episodes(
    episode_ids: list[int], config: TemporalAdapterConfig
) -> dict[str, list[int]]:
    rng = np.random.default_rng(config.split_seed)
    shuffled = np.asarray(episode_ids, dtype=np.int64)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(n * config.test_fraction)))
    n_val = max(1, int(round(n * config.val_fraction)))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    return {
        "test": sorted(int(value) for value in shuffled[:n_test]),
        "validation": sorted(int(value) for value in shuffled[n_test : n_test + n_val]),
        "train": sorted(int(value) for value in shuffled[n_test + n_val :]),
    }


def _tensor_dataset(windows: list[dict[str, Any]]) -> TensorDataset:
    return TensorDataset(
        torch.stack([row["source"] for row in windows]),
        torch.stack([row["base_delta"] for row in windows]),
        torch.stack([row["target"] for row in windows]),
        torch.stack([row["future_actions"][0] for row in windows]),
    )


def _fit_adapter(
    train_windows: list[dict[str, Any]],
    validation_windows: list[dict[str, Any]],
    mode: str,
    weight: float,
    seed: int,
    config: TemporalAdapterConfig,
) -> ResidualAdapter:
    from experiments.empirical_spectral_audit.regularizers import operator_alignment_loss

    torch.manual_seed(int(seed))
    latent_dim = int(train_windows[0]["source"].numel())
    action_dim = int(train_windows[0]["future_actions"].shape[-1])
    adapter = ResidualAdapter(latent_dim, action_dim)
    loader = DataLoader(
        _tensor_dataset(train_windows),
        batch_size=min(config.batch_size, len(train_windows)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    validation_source = torch.stack([row["source"] for row in validation_windows])
    validation_delta = torch.stack([row["base_delta"] for row in validation_windows])
    validation_target = torch.stack([row["target"] for row in validation_windows])
    best_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        adapter.train()
        for source, base_delta, target, action in loader:
            prediction = adapter(source, base_delta, action)
            prediction_loss = (prediction - target).square().mean()
            alignment_loss = operator_alignment_loss(
                source,
                prediction,
                actions=action,
                mode=mode,
                num_bins=config.alignment_bins,
                min_samples=config.alignment_min_samples,
                ridge=config.alignment_ridge,
            )
            loss = prediction_loss + float(weight) * alignment_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        adapter.eval()
        with torch.no_grad():
            validation_prediction = adapter(validation_source, validation_delta, torch.stack([
                row["future_actions"][0] for row in validation_windows
            ]))
            validation_mse = float((validation_prediction - validation_target).square().mean())
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_state = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
    if best_state is None:
        raise RuntimeError("validation did not produce an adapter checkpoint")
    adapter.load_state_dict(best_state)
    return adapter


@torch.no_grad()
def _rollout_window(
    model: torch.nn.Module,
    adapter: ResidualAdapter | None,
    row: dict[str, Any],
    max_horizon: int,
) -> dict[str, float]:
    context = row["context"].unsqueeze(0)
    context_actions = row["context_actions"].unsqueeze(0)
    targets = row["targets"]
    predictions: list[torch.Tensor] = []
    for step in range(max_horizon):
        base_next = model.predict(context, context_actions)[:, -1]
        if adapter is None:
            next_state = base_next
        else:
            current = context[:, -1]
            next_state = adapter(current, base_next - current, context_actions[:, -1])
        predictions.append(next_state[0].cpu())
        if step + 1 < max_horizon:
            next_action = row["future_actions"][step + 1].view(1, 1, -1)
            context = torch.cat((context, next_state[:, None]), dim=1)[:, -CONTEXT_LENGTH:]
            context_actions = torch.cat((context_actions, next_action), dim=1)[:, -CONTEXT_LENGTH:]
    output: dict[str, float] = {}
    for horizon in (1, 2, 4, 8):
        if horizon <= max_horizon:
            output[str(horizon)] = float((predictions[horizon - 1] - targets[horizon - 1]).square().mean())
    return output


def _bootstrap_episode(values: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, Any]:
    unique = np.asarray(sorted(set(groups.tolist())), dtype=np.int64)
    cluster_means = np.asarray([values[groups == group].mean() for group in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = cluster_means[rng.integers(0, len(cluster_means), size=(4000, len(cluster_means)))].mean(axis=1)
    return {
        "n_windows": int(values.size),
        "n_episodes": int(unique.size),
        "mean": float(cluster_means.mean()),
        "std": float(cluster_means.std(ddof=0)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_unit": "episode",
        "bootstrap_repeats": 4000,
    }


def _sign_flip(values: np.ndarray) -> float | None:
    if values.size > 12:
        return None  # type: ignore[return-value]
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return exceed / float(2 ** int(values.size))


def _evaluate(
    model: torch.nn.Module,
    adapter: ResidualAdapter | None,
    windows: list[dict[str, Any]],
    max_horizon: int,
) -> dict[str, Any]:
    rows = []
    for row in windows:
        metrics = _rollout_window(model, adapter, row, max_horizon)
        rows.append({"episode_id": row["episode_id"], "start_observation": row["start_observation"], **metrics})
    summary = {}
    groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
    for horizon in (1, 2, 4, 8):
        values = np.asarray([row[str(horizon)] for row in rows], dtype=np.float64)
        summary[str(horizon)] = _bootstrap_episode(values, groups, seed=20261003 + horizon)
    return {"rows": rows, "summary": summary}


def run_report(
    val_root: Path,
    policy: str,
    output: Path,
    device_name: str,
    normalization_mode: str,
    config: TemporalAdapterConfig,
    seeds: list[int],
    weights: list[float],
    split_seeds: list[int],
) -> dict[str, Any]:
    if not (val_root / "states.pth").exists() or not (val_root / "rel_actions.pth").exists():
        raise FileNotFoundError("public validation root lacks states.pth or rel_actions.pth")
    from experiments.io_utils import load_policy

    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, action_source = _normalization_parameters(actions, normalization_mode)
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(v) for v in pickle.load(handle)]
    episode_ids = list(range(len(lengths)))
    windows = _collect_windows(model, val_root, episode_ids, config, device, normalization_mode, action_mean, action_std)
    records: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    started = time.time()
    conditions = [("none", 0.0)]
    conditions.extend(("global", float(weight)) for weight in weights)
    conditions.extend(("conditional", float(weight)) for weight in weights)
    for split_seed in split_seeds:
        split_config = replace(config, split_seed=int(split_seed))
        split = _split_episodes(episode_ids, split_config)
        by_split = {
            name: [row for row in windows if row["episode_id"] in ids]
            for name, ids in split.items()
        }
        if any(not by_split[name] for name in ("train", "validation", "test")):
            raise ValueError(f"episode split produced an empty window split: {split}")
        split_reports[str(split_seed)] = {
            "episode_split": split,
            "frozen_base_test": _evaluate(model, None, by_split["test"], config.max_horizon),
        }
        for mode, weight in conditions:
            for seed in seeds:
                print(
                    f"[external-temporal-adapter] split={split_seed} mode={mode} weight={weight} seed={seed}: "
                    "evaluating held-out episode rollout",
                    flush=True,
                )
                adapter = _fit_adapter(by_split["train"], by_split["validation"], mode, weight, seed, config)
                evaluation = _evaluate(model, adapter, by_split["test"], config.max_horizon)
                records.append({
                    "split_seed": int(split_seed),
                    "mode": mode,
                    "alignment_weight": weight,
                    "seed": int(seed),
                    "evaluation": evaluation,
                })
    baseline_by_seed = {
        (int(row["split_seed"]), int(row["seed"])): row
        for row in records
        if row["mode"] == "none"
    }
    summary: dict[str, Any] = {}
    for mode, weight in conditions:
        rows = [row for row in records if row["mode"] == mode and float(row["alignment_weight"]) == weight]
        key = f"{mode}:{weight}"
        summary[key] = {
            "n_seeds": len(rows),
            "horizons": {},
        }
        for horizon in (1, 2, 4, 8):
            values = np.asarray([row["evaluation"]["summary"][str(horizon)]["mean"] for row in rows], dtype=np.float64)
            baseline_values = np.asarray([baseline_by_seed[(int(row["split_seed"]), int(row["seed"]))]["evaluation"]["summary"][str(horizon)]["mean"] for row in rows], dtype=np.float64)
            delta = values - baseline_values
            split_deltas = np.asarray([
                float(np.mean([
                    row["evaluation"]["summary"][str(horizon)]["mean"]
                    for row in rows
                    if int(row["split_seed"]) == split_seed
                ]))
                - float(np.mean([
                    baseline_by_seed[(split_seed, int(row["seed"]))]["evaluation"]["summary"][str(horizon)]["mean"]
                    for row in rows
                    if int(row["split_seed"]) == split_seed
                ]))
                for split_seed in split_seeds
            ], dtype=np.float64)
            split_ci = _bootstrap_episode(
                split_deltas,
                np.arange(len(split_deltas), dtype=np.int64),
                20400 + horizon,
            )
            summary[key]["horizons"][str(horizon)] = {
                "mse_mean": float(values.mean()),
                "mse_std": float(values.std(ddof=0)),
                "n_split_seeds": len(set(int(row["split_seed"]) for row in rows)),
                "run_level_delta_mean": float(delta.mean()),
                "split_level_delta_values": split_deltas.tolist(),
                "delta_vs_none_mean": float(split_deltas.mean()),
                "delta_vs_none_ci95": split_ci["ci95"],
                "exact_sign_flip_p": _sign_flip(split_deltas),
            }
    report = {
        "artifact": "external_frozen_lewm_temporal_alignment_adapter",
        "status": "complete",
        "source": {
            "validation_root": str(val_root),
            "validation_root_rel_actions_sha256": _sha256(val_root / "rel_actions.pth"),
            "policy": policy,
            "episodes": len(episode_ids),
            "windows": len(windows),
        },
        "protocol": {
            **asdict(config),
            "device": str(device),
            "normalization_mode": normalization_mode,
            "normalization_source": action_source,
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "episode_split": split,
            "conditions": conditions,
            "seeds": seeds,
            "split_seeds": [int(value) for value in split_seeds],
            "weights": weights,
            "base_encoder_and_predictor_frozen": True,
            "end_to_end_retraining": False,
            "online_control": False,
            "elapsed_seconds": time.time() - started,
        },
        "split_reports": split_reports,
        "records": records,
        "summary": summary,
        "claim_boundary": [
            "The residual adapter is trained and evaluated with episode-disjoint splits on public validation windows.",
            "The LeWM encoder and predictor remain frozen; this is not end-to-end retraining.",
            "The rollout metric is a frozen-model latent trajectory diagnostic, not online environment control or planning success.",
            "The public validation archive is external to the original training pipeline and its exact normalization lineage is recorded but not fully recoverable.",
        ],
    }
    if not _finite(report):
        raise ValueError("temporal adapter report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "Episode-disjoint frozen temporal adapter; no end-to-end or online-control claim.\n"
    )
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Frozen external PushT temporal alignment adapter",
        "",
        "A residual adapter is trained on one-step latent transitions from training episodes and evaluated autoregressively on held-out episodes. The LeWM encoder and predictor are frozen.",
        "",
        "## Episode split",
        "",
        f"- split seeds: `{report['protocol']['split_seeds']}`; windows: `{report['source']['windows']}`; adapter seeds: `{report['protocol']['seeds']}`",
        f"- normalization: `{report['protocol']['normalization_mode']}` from `{report['protocol']['normalization_source']}`",
        "- all conditions share the same encoded windows, episode split, checkpoint-selection rule, and held-out evaluation code",
        "",
        "## Held-out rollout summary",
        "",
        "| mode | weight | runs | splits | h=1 MSE | h=2 MSE | h=4 MSE | h=8 MSE | h=8 split-level delta vs none | h=8 exact sign-flip p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)
        fmt = lambda horizon: f"{values['horizons'][horizon]['mse_mean']:.6g} ± {values['horizons'][horizon]['mse_std']:.2g}"
        h8 = values["horizons"]["8"]
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | {values['horizons']['8']['n_split_seeds']} | {fmt('1')} | {fmt('2')} | {fmt('4')} | {fmt('8')} | "
            f"{h8['delta_vs_none_mean']:.6g} [{h8['delta_vs_none_ci95'][0]:.6g}, {h8['delta_vs_none_ci95'][1]:.6g}] | {h8['exact_sign_flip_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        *[f"- {item}" for item in report["claim_boundary"]],
        "- Paired inference treats episode-split seed as the independent unit; adapter initialization seeds are averaged within each split.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--action-normalization", choices=("public", "val_zscore", "raw_div100"), default="public")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[3072])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = TemporalAdapterConfig(epochs=args.epochs, windows_per_episode=args.windows_per_episode)
    report = run_report(
        val_root=args.val_root,
        policy=args.policy,
        output=args.output,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        config=config,
        seeds=[int(value) for value in args.seeds],
        weights=[float(value) for value in args.weights],
        split_seeds=[int(value) for value in args.split_seeds],
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
