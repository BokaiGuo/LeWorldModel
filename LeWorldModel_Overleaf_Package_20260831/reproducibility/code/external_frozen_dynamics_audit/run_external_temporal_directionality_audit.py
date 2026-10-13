"""Test temporal directionality of a transferred frozen latent dynamics basis.

This audit asks whether future action summaries predict future latent increments
more strongly than a one-step-lagged action summary, a current-state-only
readout, or a within-episode deranged action assignment.  The latent basis is
fit on a separate frozen export and never refit on the external windows.

The result is a cross-fitted predictive diagnostic.  It is not a causal action
effect, physical counterfactual, online-control, planning, or task-success
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_physical_dynamics_probe import (  # noqa: E402
    _fit_probe,
    _normalization_parameters,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    _load_trajectories,
)


BANDS = {"top8": (0, 8), "middle8_32": (8, 32)}
HORIZONS = (1, 4, 8)
MODES = ("future_action", "past_action", "current_state", "future_plus_state", "past_plus_state", "deranged_action")


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_basis(path: Path, name: str, dimension: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if name not in archive.files:
            raise KeyError(f"basis artifact missing {name}")
        basis = np.asarray(archive[name], dtype=np.float64)
    if basis.shape != (dimension, dimension) or not np.isfinite(basis).all():
        raise ValueError(f"invalid basis {name}: {basis.shape}")
    if not np.allclose(basis.T @ basis, np.eye(dimension), atol=1e-5):
        raise ValueError(f"basis is not orthonormal: {name}")
    return basis


def _standardized_actions(
    actions: torch.Tensor, mode: str, mean: torch.Tensor, std: torch.Tensor
) -> np.ndarray:
    scaled = actions.detach().cpu().numpy().astype(np.float64) / 100.0
    if mode == "public":
        # mean/std already contain the public loader constants.
        pass
    elif mode in {"val_zscore", "raw_div100"}:
        pass
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    return (scaled - mean.detach().cpu().numpy()) / std.detach().cpu().numpy()


def _action_features(
    normalized_actions: np.ndarray,
    episode_id: int,
    start_observation: int,
    horizon: int,
    lag: int,
) -> np.ndarray:
    """Return mean/first/last action-block summaries for a window.

    The predictor consumes blocks at observations start+2,...,start+2+h-1.
    lag=0 uses those future blocks.  lag=1 shifts the same-length summary back
    by one observation, so it is a temporal-order control with matched shape.
    """
    first = int(start_observation) + 2 - int(lag)
    blocks = []
    episode = normalized_actions[int(episode_id)]
    for offset in range(int(horizon)):
        observation = first + offset
        if observation < 0 or observation >= episode.shape[0]:
            block = np.zeros((5, episode.shape[-1]), dtype=np.float64)
        else:
            block = episode[observation : observation + 1]
            # Each external latent step corresponds to five action rows.  The
            # public data stores one action row per frame, so use the same
            # block convention as the frozen model's loader.
            begin = observation * 5
            end = begin + 5
            block = episode[begin:end] if begin < episode.shape[0] else np.empty((0, episode.shape[-1]))
            if block.shape[0] < 5:
                padded = np.zeros((5, episode.shape[-1]), dtype=np.float64)
                padded[: block.shape[0]] = block
                block = padded
        blocks.append(block.mean(axis=0))
    sequence = np.asarray(blocks, dtype=np.float64)
    return np.concatenate((sequence.mean(axis=0), sequence[0], sequence[-1]))


def _derangement(rows: list[dict[str, Any]]) -> np.ndarray:
    """Swap action sources within each episode; exactly two windows are used."""
    indices = np.arange(len(rows), dtype=np.int64)
    by_episode: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_episode.setdefault(int(row["episode_id"]), []).append(index)
    for episode, members in by_episode.items():
        if len(members) < 2:
            raise ValueError(f"episode {episode} needs at least two windows")
        for left, right in zip(members[::2], members[1::2]):
            indices[left], indices[right] = right, left
    if np.any(indices == np.arange(len(rows))):
        raise ValueError("within-episode derangement left a fixed point")
    return indices


def _r2(prediction: np.ndarray, target: np.ndarray, target_mean: np.ndarray) -> float:
    mse = float(np.square(prediction - target).mean())
    baseline = max(float(np.square(target - target_mean).mean()), 1e-12)
    return float(1.0 - mse / baseline)


def _episode_bootstrap(values: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    unique = np.asarray(sorted(set(groups.tolist())), dtype=np.int64)
    if values.ndim != 1 or values.shape != groups.shape or unique.size < 2:
        raise ValueError("episode bootstrap requires paired one-dimensional rows")
    means = np.asarray([values[groups == group].mean() for group in unique])
    rng = np.random.default_rng(int(seed))
    draws = means[rng.integers(0, means.size, size=(4000, means.size))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "n_windows": int(values.size),
        "n_episodes": int(unique.size),
        "bootstrap_unit": "episode",
        "bootstrap_repeats": 4000,
    }


def _crossfit(
    trajectories: list[dict[str, Any]],
    actions: torch.Tensor,
    mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    bases: dict[str, np.ndarray],
    ridge: float,
) -> dict[str, Any]:
    normalized = _standardized_actions(actions, mode, action_mean, action_std)
    derangement = _derangement(trajectories)
    rows: list[dict[str, Any]] = []
    episode_ids = sorted({int(row["episode_id"]) for row in trajectories})
    row_indices = {id(row): index for index, row in enumerate(trajectories)}
    for test_episode in episode_ids:
        train = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        train_indices = [row_indices[id(row)] for row in train]
        test_indices = [row_indices[id(row)] for row in test]
        for basis_name, basis in bases.items():
            for band_name, (start, end) in BANDS.items():
                train_coords = np.stack([row["true"] @ basis[:, start:end] for row in train])
                test_coords = np.stack([row["true"] @ basis[:, start:end] for row in test])
                for horizon in HORIZONS:
                    train_target = train_coords[:, horizon] - train_coords[:, 0]
                    test_target = test_coords[:, horizon] - test_coords[:, 0]
                    train_action = np.stack([
                        _action_features(normalized, row["episode_id"], row["start_observation"], horizon, lag=0)
                        for row in train
                    ])
                    test_action = np.stack([
                        _action_features(normalized, row["episode_id"], row["start_observation"], horizon, lag=0)
                        for row in test
                    ])
                    train_past = np.stack([
                        _action_features(normalized, row["episode_id"], row["start_observation"], horizon, lag=1)
                        for row in train
                    ])
                    test_past = np.stack([
                        _action_features(normalized, row["episode_id"], row["start_observation"], horizon, lag=1)
                        for row in test
                    ])
                    train_state = train_coords[:, 0]
                    test_state = test_coords[:, 0]
                    train_deranged = np.stack([
                        _action_features(
                            normalized,
                            trajectories[int(derangement[index])]["episode_id"],
                            trajectories[int(derangement[index])]["start_observation"],
                            horizon,
                            lag=0,
                        )
                        for index in train_indices
                    ])
                    test_deranged = np.stack([
                        _action_features(
                            normalized,
                            trajectories[int(derangement[index])]["episode_id"],
                            trajectories[int(derangement[index])]["start_observation"],
                            horizon,
                            lag=0,
                        )
                        for index in test_indices
                    ])
                    feature_sets = {
                        "future_action": (train_action, test_action),
                        "past_action": (train_past, test_past),
                        "current_state": (train_state, test_state),
                        "future_plus_state": (np.concatenate([train_action, train_state], axis=1), np.concatenate([test_action, test_state], axis=1)),
                        "past_plus_state": (np.concatenate([train_past, train_state], axis=1), np.concatenate([test_past, test_state], axis=1)),
                        "deranged_action": (train_deranged, test_deranged),
                    }
                    target_mean = train_target.mean(axis=0)
                    for feature_name, (train_features, test_features) in feature_sets.items():
                        fit = _fit_probe(train_features, train_target, ridge=ridge)
                        prediction = (test_features - fit["source_mean"]) @ fit["weights"] + fit["target_mean"]
                        score = _r2(prediction, test_target, target_mean)
                        for local_index, row in enumerate(test):
                            rows.append({
                                "episode_id": int(row["episode_id"]),
                                "start_observation": int(row["start_observation"]),
                                "basis": basis_name,
                                "band": band_name,
                                "horizon": int(horizon),
                                "feature": feature_name,
                                "r2": float(score),
                                "sample_r2": _r2(prediction[local_index : local_index + 1], test_target[local_index : local_index + 1], target_mean),
                            })

    summary: dict[str, Any] = {}
    def values(feature: str, basis: str, band: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        selected = [row for row in rows if row["feature"] == feature and row["basis"] == basis and row["band"] == band and row["horizon"] == horizon]
        return np.asarray([row["sample_r2"] for row in selected]), np.asarray([row["episode_id"] for row in selected])

    for basis_name in bases:
        for band_name in BANDS:
            for horizon in HORIZONS:
                key = f"{basis_name}/{band_name}/h{horizon}"
                summary[key] = {}
                for feature in MODES:
                    sample, groups = values(feature, basis_name, band_name, horizon)
                    summary[key][feature] = _episode_bootstrap(sample, groups, 20261040 + horizon)
                summary[key]["future_plus_state_minus_past_plus_state"] = _episode_bootstrap(
                    values("future_plus_state", basis_name, band_name, horizon)[0] - values("past_plus_state", basis_name, band_name, horizon)[0],
                    values("future_plus_state", basis_name, band_name, horizon)[1],
                    20261140 + horizon,
                )
                summary[key]["future_action_minus_deranged_action"] = _episode_bootstrap(
                    values("future_action", basis_name, band_name, horizon)[0] - values("deranged_action", basis_name, band_name, horizon)[0],
                    values("future_action", basis_name, band_name, horizon)[1],
                    20261240 + horizon,
                )
                future_past = summary[key]["future_plus_state_minus_past_plus_state"]
                future_deranged = summary[key]["future_action_minus_deranged_action"]
                if future_past["ci95"][0] > 0.0 and future_deranged["ci95"][0] > 0.0:
                    verdict = "supported_temporal_directionality"
                elif future_past["mean"] > 0.0 and future_deranged["mean"] > 0.0:
                    verdict = "point_direction_only"
                else:
                    verdict = "not_supported"
                summary[key]["directionality_verdict"] = verdict
    return {"rows": rows, "summary": summary, "derangement": derangement.tolist()}


def run_report(
    val_root: Path,
    basis_path: Path,
    output: Path,
    policy: str,
    device_name: str,
    windows_per_episode: int,
    max_horizon: int,
    ridge: float,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    trajectories = _load_trajectories(model, val_root, device, "public", windows_per_episode, max_horizon)
    dimension = int(trajectories[0]["true"].shape[-1])
    bases = {
        "covariance_basis": _load_basis(basis_path, "covariance_basis", dimension),
        "dynamics_basis": _load_basis(basis_path, "dynamics_basis", dimension),
    }
    reports = {}
    for mode in ("public", "val_zscore", "raw_div100"):
        mean, std, source = _normalization_parameters(actions, mode)
        reports[mode] = _crossfit(trajectories, actions, mode, mean, std, bases, ridge)
        reports[mode]["normalization_source"] = source
    all_cells = [
        summary
        for value in reports.values()
        for summary in value["summary"].values()
    ]
    report = {
        "artifact": "external_frozen_lewm_temporal_directionality_audit",
        "status": "complete",
        "source": {
            "validation_root": str(val_root),
            "rel_actions_sha256": _sha256(val_root / "rel_actions.pth"),
            "basis_artifact": str(basis_path),
            "episodes": len({int(row["episode_id"]) for row in trajectories}),
            "windows": len(trajectories),
            "latent_dim": dimension,
        },
        "protocol": {
            "policy": policy,
            "device": str(device),
            "windows_per_episode": int(windows_per_episode),
            "max_horizon": int(max_horizon),
            "horizons": list(HORIZONS),
            "bands": BANDS,
            "basis_fit": "independent frozen latent export; no external refitting",
            "crossfit": "leave-one-episode-out",
            "future_action": "action blocks aligned to the latent transition",
            "past_action": "same-length action summary shifted back by one latent step",
            "deranged_action": "within-episode action-source swap preserving action marginal",
            "readout": "ridge prediction of external true latent increment in transferred basis coordinates",
            "ridge": float(ridge),
            "normalizations": ["public", "val_zscore", "raw_div100"],
        },
        "reports": reports,
        "claim_matrix": {
            "total_cells": len(all_cells),
            "supported_temporal_directionality_cells": sum(
                cell["directionality_verdict"] == "supported_temporal_directionality"
                for cell in all_cells
            ),
            "point_direction_only_cells": sum(
                cell["directionality_verdict"] == "point_direction_only"
                for cell in all_cells
            ),
            "not_supported_cells": sum(
                cell["directionality_verdict"] == "not_supported"
                for cell in all_cells
            ),
            "verdict": "not_normalization_invariant" if any(
                cell["directionality_verdict"] == "not_supported" for cell in all_cells
            ) else "supported",
        },
        "claim_boundary": [
            "Future-versus-past comparisons are temporal predictive controls, not causal action effects.",
            "The target is an encoded latent increment; it is not a physical counterfactual or task-level outcome.",
            "The transferred bases are fitted on a separate frozen export and are not refit on the external windows.",
            "All inference is clustered at the episode level; no windows are treated as independent episodes.",
        ],
    }
    if not _finite(report):
        raise ValueError("directionality report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text("Frozen external future-vs-past action directionality diagnostic; no causal-control claim.\n")
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# External frozen temporal directionality audit",
        "",
        "The transferred basis is evaluated with episode-disjoint future-action, one-step-lagged past-action, current-state, and within-episode deranged-action controls. Positive future-minus-past values favor temporally aligned action information.",
        "",
        "| normalization | basis | band | h | future+state R2 | past+state R2 | future-past delta | future-deranged delta | verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for normalization, value in report["reports"].items():
        for key, summary in value["summary"].items():
            basis, band, horizon = key.split("/")
            future = summary["future_plus_state"]
            past = summary["past_plus_state"]
            delta = summary["future_plus_state_minus_past_plus_state"]
            deranged = summary["future_action_minus_deranged_action"]
            lines.append(
                f"| {normalization} | {basis} | {band} | {horizon[1:]} | {future['mean']:.6g} | {past['mean']:.6g} | "
                f"{delta['mean']:.6g} [{delta['ci95'][0]:.6g}, {delta['ci95'][1]:.6g}] | "
                f"{deranged['mean']:.6g} [{deranged['ci95'][0]:.6g}, {deranged['ci95'][1]:.6g}] | {summary['directionality_verdict']} |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run_report(args.val_root, args.basis, args.output, args.policy, args.device, args.windows_per_episode, args.max_horizon, args.ridge)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
