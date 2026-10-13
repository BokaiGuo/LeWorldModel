"""Stress-test cross-horizon latent readouts against a richer action history.

The existing cross-horizon audit uses a compact six-dimensional action
summary.  This companion audit keeps the source-fitted latent bases,
leave-one-episode-out folds, horizon transfer, ridge convention, and
episode-cluster bootstrap fixed, but compares that summary with the complete
sequence of per-block action means (2*h dimensions).  It is designed to test
whether latent incremental R2 survives a less restrictive observed-action
control.  The artifact remains a frozen predictive diagnostic, not a causal
action-effect, physical-counterfactual, planning, or task-success result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_cross_horizon_readout import (  # noqa: E402
    _state_features,
)
from experiments.empirical_spectral_audit.run_external_physical_dynamics_probe import (  # noqa: E402
    _action_summary,
)
from experiments.empirical_spectral_audit.run_external_physical_observability_audit import (  # noqa: E402
    _state_targets,
)
from experiments.empirical_spectral_audit.run_external_rank_target_specificity_audit import (  # noqa: E402
    HORIZONS,
    RANKS,
    TARGET_NAMES,
    _decode,
    _finite,
    _fit_probe,
    _load_all_bases,
    _r2,
    _target,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _bootstrap_clustered,
    _load_trajectories,
    _normalization_parameters,
)
from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    _standardize_public_actions,
)


ACTION_CONTROLS = (
    "compact6",
    "block_mean_history_padded",
    "normalized_time_dct16",
)
MAX_ACTION_HORIZON = max(HORIZONS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_mean_history(
    raw_actions: torch.Tensor,
    episode_id: int,
    start_observation: int,
    horizon: int,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> np.ndarray:
    normalized = _standardize_public_actions(
        raw_actions[episode_id],
        normalization_mode=normalization_mode,
        action_mean=action_mean,
        action_std=action_std,
    )
    blocks: list[np.ndarray] = []
    first_obs = int(start_observation) + CONTEXT_LENGTH - 1
    for step in range(int(horizon)):
        obs = first_obs + step
        block = normalized[obs * FRAME_STRIDE : (obs + 1) * FRAME_STRIDE]
        if block.shape[0] < FRAME_STRIDE:
            block = torch.cat(
                [
                    block,
                    torch.zeros(
                        (FRAME_STRIDE - block.shape[0], 2), dtype=block.dtype
                    ),
                ]
            )
        blocks.append(block.mean(dim=0).numpy())
    # Cross-horizon transfer requires a common feature dimension.  The
    # observed block means are left-aligned and the unobserved suffix is
    # explicitly zero-padded to the prespecified maximum horizon.
    padded = np.zeros((MAX_ACTION_HORIZON, 2), dtype=np.float64)
    padded[: int(horizon)] = np.asarray(blocks, dtype=np.float64)
    return padded.reshape(-1)


def _normalized_time_dct16(
    raw_actions: torch.Tensor,
    episode_id: int,
    start_observation: int,
    horizon: int,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> np.ndarray:
    """Encode the complete observed action sequence on a common time grid.

    The interpolation is only along the declared within-window temporal
    coordinate; it never uses action values outside the requested horizon.
    The orthonormal DCT makes the representation fixed-dimensional while
    retaining low- and high-frequency temporal shape components.
    """
    normalized = _standardize_public_actions(
        raw_actions[episode_id],
        normalization_mode=normalization_mode,
        action_mean=action_mean,
        action_std=action_std,
    )
    blocks: list[np.ndarray] = []
    first_obs = int(start_observation) + CONTEXT_LENGTH - 1
    for step in range(int(horizon)):
        obs = first_obs + step
        block = normalized[obs * FRAME_STRIDE : (obs + 1) * FRAME_STRIDE]
        if block.shape[0] < FRAME_STRIDE:
            block = torch.cat(
                [
                    block,
                    torch.zeros(
                        (FRAME_STRIDE - block.shape[0], 2), dtype=block.dtype
                    ),
                ]
            )
        blocks.append(block.mean(dim=0).numpy())
    sequence = np.asarray(blocks, dtype=np.float64)
    source_time = np.linspace(0.0, 1.0, int(horizon))
    target_time = np.linspace(0.0, 1.0, MAX_ACTION_HORIZON)
    resampled = np.stack(
        [
            np.interp(target_time, source_time, sequence[:, coordinate])
            for coordinate in range(sequence.shape[1])
        ],
        axis=1,
    )
    index = np.arange(MAX_ACTION_HORIZON, dtype=np.float64)
    dct = np.sqrt(2.0 / MAX_ACTION_HORIZON) * np.cos(
        np.pi
        * (index[:, None] + 0.5)
        * index[None, :]
        / MAX_ACTION_HORIZON
    )
    dct[0] /= np.sqrt(2.0)
    return (dct @ resampled).reshape(-1)


def _action_features(
    raw_actions: torch.Tensor,
    episode_id: int,
    start_observation: int,
    horizon: int,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    control: str,
) -> np.ndarray:
    if control == "compact6":
        return _action_summary(
            raw_actions,
            episode_id,
            start_observation,
            horizon,
            normalization_mode,
            action_mean,
            action_std,
        )
    if control == "block_mean_history_padded":
        return _block_mean_history(
            raw_actions,
            episode_id,
            start_observation,
            horizon,
            normalization_mode,
            action_mean,
            action_std,
        )
    if control == "normalized_time_dct16":
        return _normalized_time_dct16(
            raw_actions,
            episode_id,
            start_observation,
            horizon,
            normalization_mode,
            action_mean,
            action_std,
        )
    raise ValueError(f"unknown action control: {control}")


def _regularized_condition(source: np.ndarray, ridge: float) -> float:
    centered = np.asarray(source, dtype=np.float64) - np.asarray(source).mean(axis=0)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / max(source.shape[1], 1), 1e-8)
    regularized = covariance + float(ridge) * scale * np.eye(source.shape[1])
    return float(np.linalg.cond(regularized))


def _crossfit(
    trajectories: list[dict[str, Any]],
    actions: torch.Tensor,
    mean: np.ndarray,
    bases: dict[str, np.ndarray],
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    ridge: float,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    episode_ids = sorted({int(row["episode_id"]) for row in trajectories})
    action_cache = {
        (id(row), control, horizon): _action_features(
            actions,
            int(row["episode_id"]),
            int(row["start_observation"]),
            horizon,
            normalization_mode,
            action_mean,
            action_std,
            control,
        )
        for row in trajectories
        for control in ACTION_CONTROLS
        for horizon in HORIZONS
    }
    records: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    for test_episode in episode_ids:
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        for source_horizon in HORIZONS:
            for control in ACTION_CONTROLS:
                train_action = np.stack(
                    [action_cache[(id(row), control, source_horizon)] for row in train_rows]
                )
                condition_records.append(
                    {
                        "test_episode": int(test_episode),
                        "source_horizon": int(source_horizon),
                        "control": control,
                        "feature_dim": int(train_action.shape[1]),
                        "train_rows": int(train_action.shape[0]),
                        "regularized_condition_number": _regularized_condition(
                            train_action, ridge
                        ),
                    }
                )
                for target_horizon in HORIZONS:
                    test_action = np.stack(
                        [action_cache[(id(row), control, target_horizon)] for row in test_rows]
                    )
                    for target_name in TARGET_NAMES:
                        train_target = np.stack(
                            [
                                _target(row, source_horizon, target_name, bases)
                                for row in train_rows
                            ]
                        )
                        test_target = np.stack(
                            [
                                _target(row, target_horizon, target_name, bases)
                                for row in test_rows
                            ]
                        )
                        target_mean = train_target.mean(axis=0)
                        action_probe = _fit_probe(train_action, train_target, ridge)
                        action_prediction = _decode(action_probe, test_action)
                        action_values = [
                            _r2(prediction, target, target_mean)
                            for prediction, target in zip(action_prediction, test_target)
                        ]
                        for basis_name, basis in bases.items():
                            for rank in RANKS:
                                train_state = _state_features(train_rows, mean, basis, rank)
                                test_state = _state_features(test_rows, mean, basis, rank)
                                state_probe = _fit_probe(train_state, train_target, ridge)
                                joint_probe = _fit_probe(
                                    np.concatenate([train_action, train_state], axis=1),
                                    train_target,
                                    ridge,
                                )
                                state_prediction = _decode(state_probe, test_state)
                                joint_prediction = _decode(
                                    joint_probe,
                                    np.concatenate([test_action, test_state], axis=1),
                                )
                                for row, action_r2, state_pred, joint_pred, target in zip(
                                    test_rows,
                                    action_values,
                                    state_prediction,
                                    joint_prediction,
                                    test_target,
                                ):
                                    joint_r2 = _r2(joint_pred, target, target_mean)
                                    records.append(
                                        {
                                            "episode_id": int(test_episode),
                                            "start_observation": int(row["start_observation"]),
                                            "source_horizon": int(source_horizon),
                                            "target_horizon": int(target_horizon),
                                            "target": target_name,
                                            "basis": basis_name,
                                            "rank": int(rank),
                                            "action_control": control,
                                            "action_r2": float(action_r2),
                                            "joint_r2": float(joint_r2),
                                            "state_incremental_r2_over_action": float(
                                                joint_r2 - action_r2
                                            ),
                                        }
                                    )

    metrics = ("action_r2", "joint_r2", "state_incremental_r2_over_action")
    summary: dict[str, Any] = {}
    for control in ACTION_CONTROLS:
        for source_horizon in HORIZONS:
            for target_horizon in HORIZONS:
                for target_name in TARGET_NAMES:
                    for basis_name in bases:
                        for rank in RANKS:
                            rows = [
                                row
                                for row in records
                                if row["action_control"] == control
                                and row["source_horizon"] == source_horizon
                                and row["target_horizon"] == target_horizon
                                and row["target"] == target_name
                                and row["basis"] == basis_name
                                and row["rank"] == rank
                            ]
                            groups = np.asarray(
                                [row["episode_id"] for row in rows], dtype=np.int64
                            )
                            value: dict[str, Any] = {"windows": len(rows)}
                            for offset, metric in enumerate(metrics):
                                value[metric] = _bootstrap_clustered(
                                    np.asarray([row[metric] for row in rows]),
                                    groups,
                                    bootstrap_repeats,
                                    20262000
                                    + offset * 100000
                                    + source_horizon * 10000
                                    + target_horizon * 1000
                                    + rank
                                    + (0 if control == "compact6" else 5000000),
                                )
                            summary[
                                f"{control}/h{source_horizon}_to_h{target_horizon}/"
                                f"{target_name}/{basis_name}/k{rank}"
                            ] = value

    indexed = {
        (
            row["episode_id"],
            row["start_observation"],
            row["source_horizon"],
            row["target_horizon"],
            row["target"],
            row["basis"],
            row["rank"],
            row["action_control"],
        ): row
        for row in records
    }
    contrasts: dict[str, Any] = {}
    basis_pairs = (
        ("dynamics_basis", "covariance_basis", "dynamics_minus_covariance"),
        ("dynamics_basis", "random_basis", "dynamics_minus_random"),
        ("residual_basis", "dynamics_basis", "residual_minus_dynamics"),
    )
    for control in ACTION_CONTROLS:
        for source_horizon in HORIZONS:
            for target_horizon in HORIZONS:
                for target_name in TARGET_NAMES:
                    for rank in RANKS:
                        for left, right, contrast_name in basis_pairs:
                            paired, groups = [], []
                            for row in records:
                                if (
                                    row["action_control"] != control
                                    or row["source_horizon"] != source_horizon
                                    or row["target_horizon"] != target_horizon
                                    or row["target"] != target_name
                                    or row["basis"] != left
                                    or row["rank"] != rank
                                ):
                                    continue
                                right_row = indexed[
                                    (
                                        row["episode_id"],
                                        row["start_observation"],
                                        source_horizon,
                                        target_horizon,
                                        target_name,
                                        right,
                                        rank,
                                        control,
                                    )
                                ]
                                paired.append(
                                    row["state_incremental_r2_over_action"]
                                    - right_row["state_incremental_r2_over_action"]
                                )
                                groups.append(row["episode_id"])
                            contrasts[
                                f"{control}/h{source_horizon}_to_h{target_horizon}/"
                                f"{target_name}/k{rank}/{contrast_name}"
                            ] = _bootstrap_clustered(
                                np.asarray(paired, dtype=np.float64),
                                np.asarray(groups, dtype=np.int64),
                                bootstrap_repeats,
                                20263000
                                + source_horizon * 10000
                                + target_horizon * 1000
                                + rank
                                + (0 if control == "compact6" else 5000000),
                            )

    control_contrasts: dict[str, Any] = {}
    block_history_contrasts: dict[str, Any] = {}
    for richer_control in ACTION_CONTROLS[1:]:
        for source_horizon in HORIZONS:
            for target_horizon in HORIZONS:
                for target_name in TARGET_NAMES:
                    for basis_name in bases:
                        for rank in RANKS:
                            paired, groups = [], []
                            for row in records:
                                if (
                                    row["action_control"] != richer_control
                                    or row["source_horizon"] != source_horizon
                                    or row["target_horizon"] != target_horizon
                                    or row["target"] != target_name
                                    or row["basis"] != basis_name
                                    or row["rank"] != rank
                                ):
                                    continue
                                compact = indexed[
                                    (
                                        row["episode_id"],
                                        row["start_observation"],
                                        source_horizon,
                                        target_horizon,
                                        target_name,
                                        basis_name,
                                        rank,
                                        "compact6",
                                    )
                                ]
                                paired.append(
                                    row["state_incremental_r2_over_action"]
                                    - compact["state_incremental_r2_over_action"]
                                )
                                groups.append(row["episode_id"])
                            value = _bootstrap_clustered(
                                np.asarray(paired, dtype=np.float64),
                                np.asarray(groups, dtype=np.int64),
                                bootstrap_repeats,
                                20264000
                                + source_horizon * 10000
                                + target_horizon * 1000
                                + rank
                                + (0 if richer_control == "block_mean_history_padded" else 7000000),
                            )
                            key = (
                                f"{richer_control}/h{source_horizon}_to_h{target_horizon}/"
                                f"{target_name}/{basis_name}/k{rank}/rich_minus_compact"
                            )
                            control_contrasts[key] = value
                            if richer_control == "block_mean_history_padded":
                                block_history_contrasts[
                                    f"h{source_horizon}_to_h{target_horizon}/"
                                    f"{target_name}/{basis_name}/k{rank}/rich_minus_compact"
                                ] = value

    condition_summary: dict[str, Any] = {}
    for control in ACTION_CONTROLS:
        for horizon in HORIZONS:
            values = [
                row["regularized_condition_number"]
                for row in condition_records
                if row["control"] == control and row["source_horizon"] == horizon
            ]
            condition_summary[f"{control}/h{horizon}"] = {
                "feature_dim": int(
                    next(row["feature_dim"] for row in condition_records if row["control"] == control and row["source_horizon"] == horizon)
                ),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
    return {
        "records": records,
        "summary": summary,
        "paired_contrasts": contrasts,
        "rich_minus_compact": block_history_contrasts,
        "control_minus_compact": control_contrasts,
        "condition_records": condition_records,
        "condition_summary": condition_summary,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]
    contrasts = report["crossfit"]["paired_contrasts"]
    control_contrasts = report["crossfit"]["rich_minus_compact"]
    lines = [
        "# Cross-horizon readout under richer observed-action history",
        "",
        "The latent bases, leave-one-episode-out folds, ridge convention, and episode-cluster bootstrap are fixed. `compact6` is the legacy six-dimensional action summary; `block_mean_history_padded` retains the complete observed sequence with zero-padding; `normalized_time_dct16` retains the complete sequence after normalized-time resampling and an orthonormal DCT. Positive incremental R2 is predictive association beyond the declared action control, not a causal action effect.",
        "",
        "## Dynamics-band target, top-8 input",
        "",
        "| control | train -> test | dynamics | covariance | residual | random | dynamics-random CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for control in ACTION_CONTROLS:
        for source_horizon in HORIZONS:
            for target_horizon in HORIZONS:
                values = []
                for basis_name in ("dynamics_basis", "covariance_basis", "residual_basis", "random_basis"):
                    key = f"{control}/h{source_horizon}_to_h{target_horizon}/dynamics_band_increment/{basis_name}/k8"
                    values.append(summary[key]["state_incremental_r2_over_action"]["mean"])
                contrast = contrasts[f"{control}/h{source_horizon}_to_h{target_horizon}/dynamics_band_increment/k8/dynamics_minus_random"]
                lines.append(
                    f"| {control} | h{source_horizon} -> h{target_horizon} | "
                    + " | ".join(f"{value:+.4f}" for value in values)
                    + f" | {contrast['mean']:+.4f} [{contrast['ci95'][0]:+.4f},{contrast['ci95'][1]:+.4f}] |"
                )
    lines.extend(["", "## Richer-control change", "", "Positive values mean that retaining the complete block-mean history increases the reported latent incremental R2 relative to compact6.", "", "| train -> test | basis | rich minus compact |", "| --- | --- | ---: |"])
    for source_horizon, target_horizon in ((1, 4), (1, 8), (4, 8)):
        for basis_name in ("dynamics_basis", "covariance_basis", "residual_basis", "random_basis"):
            key = f"h{source_horizon}_to_h{target_horizon}/dynamics_band_increment/{basis_name}/k8/rich_minus_compact"
            value = control_contrasts[key]
            lines.append(f"| h{source_horizon} -> h{target_horizon} | {basis_name} | {value['mean']:+.4f} [{value['ci95'][0]:+.4f},{value['ci95'][1]:+.4f}] |")
    lines.extend(["", "## Conditioning diagnostics", "", "| control | horizon | feature dimension | regularized condition number (min / median / max) |", "| --- | ---: | ---: | ---: |"])
    for control in ACTION_CONTROLS:
        for horizon in HORIZONS:
            value = report["crossfit"]["condition_summary"][f"{control}/h{horizon}"]
            lines.append(f"| {control} | h{horizon} | {value['feature_dim']} | {value['min']:.3g} / {value['median']:.3g} / {value['max']:.3g} |")
    lines.extend(["", "## Interpretation boundary", "", "- The richer control is still an observed-action summary and does not identify a causal action effect.", "- The result supports or weakens only the claim that latent predictive information remains after conditioning on this declared action-history representation.", "- It does not establish physical counterfactual prediction, online planning, task success, retraining improvement, or a universal world-model property."])
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    dynamics_basis_path: Path,
    residual_basis_path: Path,
    output: Path,
    policy: str = "pusht/lewm",
    device_name: str = "cuda",
    normalization_mode: str = "public",
    windows_per_episode: int = 2,
    max_horizon: int = 8,
    random_seed: int = 20261003,
    ridge: float = 1e-2,
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, action_source = _normalization_parameters(actions, normalization_mode)
    trajectories = _load_trajectories(model, val_root, device, normalization_mode, windows_per_episode, max_horizon)
    states = torch.load(val_root / "states.pth", map_location="cpu", weights_only=False).float()
    trajectories = _state_targets(trajectories, states, max_horizon)
    mean, bases = _load_all_bases(dynamics_basis_path, residual_basis_path, random_seed)
    crossfit = _crossfit(trajectories, actions, mean, bases, normalization_mode, action_mean, action_std, ridge, bootstrap_repeats)
    report = {
        "artifact": "external_frozen_lewm_cross_horizon_action_control_robustness",
        "status": "complete",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "dynamics_basis_artifact": str(dynamics_basis_path),
            "dynamics_basis_sha256": _sha256(dynamics_basis_path),
            "residual_basis_artifact": str(residual_basis_path),
            "residual_basis_sha256": _sha256(residual_basis_path),
            "episodes": int(len({row["episode_id"] for row in trajectories})),
            "windows": int(len(trajectories)),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": list(HORIZONS),
            "ranks": list(RANKS),
            "bases": list(bases),
            "targets": list(TARGET_NAMES),
            "action_controls": {
                "compact6": "mean action sequence plus first and last block means",
                "block_mean_history_padded": "left-aligned complete sequence of per-block two-dimensional action means, zero-padded to h=8",
                "normalized_time_dct16": "complete per-block action-mean sequence resampled to normalized time and encoded by an orthonormal 16-dimensional DCT",
            },
            "crossfit": "leave-one-episode-out",
            "transfer": "fit at source horizon, evaluate at target horizon",
            "action_normalization": {"mode": normalization_mode, "raw_action_divisor": 100.0, "mean": action_mean.tolist(), "std": action_std.tolist(), "source": action_source},
            "ridge": float(ridge),
            "bootstrap": "episode-cluster bootstrap",
            "bootstrap_repeats": int(bootstrap_repeats),
            "random_basis_seed": int(random_seed),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "The richer action-history control retains every per-block action mean but remains an observed-action predictive baseline.",
            "The latent bases are fitted on a separate frozen export and are never refit on external validation windows.",
            "Positive incremental R2 is predictive association beyond the declared action history, not a causal action effect.",
            "The experiment does not establish physical causality, counterfactual prediction, online planning, task success, retraining improvement, or a universal world model.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("action_control_notes.txt").write_text("Richer observed-action history control: compact6 versus complete per-block action means; frozen latent readout and episode-level inference.\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dynamics-basis", type=Path, required=True)
    parser.add_argument("--residual-basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20261003)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        archive=args.archive,
        dynamics_basis_path=args.dynamics_basis,
        residual_basis_path=args.residual_basis,
        output=args.output,
        policy=args.policy,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        windows_per_episode=args.windows_per_episode,
        max_horizon=args.max_horizon,
        random_seed=args.random_seed,
        ridge=args.ridge,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
