"""Audit rank-specific and target-specific information in frozen latent bases.

The audit uses source-fitted frozen bases and leave-one-episode-out probes on
external PushT windows.  For every rank in a prespecified curve it compares
the added predictive value of current latent coordinates beyond observed
actions for two targets: encoded latent displacement and physical displacement.

This is a frozen predictive diagnostic.  It is not a causal action effect,
physical counterfactual, online-planning, or task-success evaluation.
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

from experiments.empirical_spectral_audit.run_external_physical_band_probe import (  # noqa: E402
    _random_basis,
    _band_coordinates,
)
from experiments.empirical_spectral_audit.run_external_physical_dynamics_probe import (  # noqa: E402
    _action_summary,
    _wrap_angle,
)
from experiments.empirical_spectral_audit.run_external_physical_observability_audit import (  # noqa: E402
    STATE_NAMES,
    _state_targets,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _bootstrap_clustered,
    _load_trajectories,
    _normalization_parameters,
)


HORIZONS = (1, 4, 8)
RANKS = (1, 2, 4, 8, 16, 32)
BASIS_NAMES = ("covariance_basis", "dynamics_basis", "residual_basis", "random_basis")
TARGET_NAMES = (
    "dynamics_band_increment",
    "residual_band_increment",
    "full_latent_increment",
    "physical_displacement",
)


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


def _fit_probe(source: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered = source - source_mean
    covariance = centered.T @ centered / max(source.shape[0] - 1, 1)
    cross = centered.T @ (target - target_mean) / max(source.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / max(source.shape[1], 1), 1e-8)
    weights = np.linalg.solve(
        covariance + float(ridge) * scale * np.eye(source.shape[1]), cross
    )
    return {"source_mean": source_mean, "target_mean": target_mean, "weights": weights}


def _decode(probe: dict[str, np.ndarray], source: np.ndarray) -> np.ndarray:
    return (source - probe["source_mean"]) @ probe["weights"] + probe["target_mean"]


def _r2(prediction: np.ndarray, target: np.ndarray, target_mean: np.ndarray) -> float:
    mse = float(np.square(prediction - target).mean())
    variance = max(float(np.square(target - target_mean).mean()), 1e-12)
    return float(1.0 - mse / variance)


def _physical_delta(row: dict[str, Any], horizon: int) -> np.ndarray:
    delta = np.asarray(row["physical"][horizon] - row["physical"][0], dtype=np.float64)
    return _wrap_angle(delta[None])[0]


def _load_all_bases(dynamics_basis_path: Path, residual_basis_path: Path, random_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(dynamics_basis_path, allow_pickle=False) as dynamics_archive:
        required_dynamics = {"latent_mean", "covariance_basis", "dynamics_basis"}
        missing = sorted(required_dynamics.difference(dynamics_archive.files))
        if missing:
            raise KeyError(f"dynamics basis artifact missing fields: {missing}")
        mean = np.asarray(dynamics_archive["latent_mean"], dtype=np.float64)
        covariance = np.asarray(dynamics_archive["covariance_basis"], dtype=np.float64)
        dynamics = np.asarray(dynamics_archive["dynamics_basis"], dtype=np.float64)
    with np.load(residual_basis_path, allow_pickle=False) as residual_archive:
        required_residual = {
            "latent_mean",
            "residual_basis",
            "residual_covariance_basis",
        }
        missing = sorted(required_residual.difference(residual_archive.files))
        if missing:
            raise KeyError(f"residual basis artifact missing fields: {missing}")
        residual_mean = np.asarray(residual_archive["latent_mean"], dtype=np.float64)
        residual = np.asarray(residual_archive["residual_basis"], dtype=np.float64)
        residual_covariance = np.asarray(
            residual_archive["residual_covariance_basis"], dtype=np.float64
        )
    if residual_mean.shape != mean.shape or not np.allclose(residual_mean, mean, atol=1e-10):
        raise ValueError("dynamics and residual basis artifacts use different latent means")
    dimension = int(mean.size)
    ordered = {
        "covariance_basis": covariance,
        "dynamics_basis": dynamics,
        "residual_basis": residual,
        "random_basis": _random_basis(dimension, random_seed),
        "residual_covariance_basis": residual_covariance,
    }
    for name, basis in ordered.items():
        if basis.shape != (dimension, dimension) or not np.isfinite(basis).all():
            raise ValueError(f"invalid basis shape or finite values: {name}")
        if not np.allclose(basis.T @ basis, np.eye(dimension), atol=1e-5):
            raise ValueError(f"basis is not orthonormal: {name}")
    return mean, ordered


def _target(
    row: dict[str, Any],
    horizon: int,
    target_name: str,
    bases: dict[str, np.ndarray],
) -> np.ndarray:
    increment = np.asarray(row["true"][horizon] - row["true"][0], dtype=np.float64)
    if target_name == "dynamics_band_increment":
        return increment @ bases["dynamics_basis"][:, :8]
    if target_name == "residual_band_increment":
        return increment @ bases["residual_basis"][:, :8]
    if target_name == "full_latent_increment":
        return increment
    if target_name == "physical_displacement":
        return _physical_delta(row, horizon)
    raise ValueError(f"unknown target {target_name}")


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
        (id(row), horizon): _action_summary(
            actions,
            int(row["episode_id"]),
            int(row["start_observation"]),
            horizon,
            normalization_mode,
            action_mean,
            action_std,
        )
        for row in trajectories
        for horizon in HORIZONS
    }
    records: list[dict[str, Any]] = []
    for test_episode in episode_ids:
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        for horizon in HORIZONS:
            train_action = np.stack([action_cache[(id(row), horizon)] for row in train_rows])
            test_action = np.stack([action_cache[(id(row), horizon)] for row in test_rows])
            for target_name in TARGET_NAMES:
                train_target = np.stack(
                    [_target(row, horizon, target_name, bases) for row in train_rows]
                )
                test_target = np.stack(
                    [_target(row, horizon, target_name, bases) for row in test_rows]
                )
                target_mean = train_target.mean(axis=0)
                action_probe = _fit_probe(train_action, train_target, ridge)
                for basis_name, basis in bases.items():
                    for rank in RANKS:
                        train_state = np.stack([
                            _band_coordinates(row["true"], mean, basis, (0, rank))[0]
                            for row in train_rows
                        ])
                        test_state = np.stack([
                            _band_coordinates(row["true"], mean, basis, (0, rank))[0]
                            for row in test_rows
                        ])
                        state_probe = _fit_probe(train_state, train_target, ridge)
                        joint_probe = _fit_probe(
                            np.concatenate([train_action, train_state], axis=1),
                            train_target,
                            ridge,
                        )
                        action_prediction = _decode(action_probe, test_action)
                        state_prediction = _decode(state_probe, test_state)
                        joint_prediction = _decode(
                            joint_probe, np.concatenate([test_action, test_state], axis=1)
                        )
                        for row, action_pred, state_pred, joint_pred in zip(
                            test_rows, action_prediction, state_prediction, joint_prediction
                        ):
                            target = _target(row, horizon, target_name, bases)
                            records.append(
                                {
                                    "episode_id": int(test_episode),
                                    "start_observation": int(row["start_observation"]),
                                    "target": target_name,
                                    "basis": basis_name,
                                    "rank": int(rank),
                                    "horizon": int(horizon),
                                    "action_r2": _r2(action_pred, target, target_mean),
                                    "state_r2": _r2(state_pred, target, target_mean),
                                    "joint_r2": _r2(joint_pred, target, target_mean),
                                    "state_incremental_r2_over_action": (
                                        _r2(joint_pred, target, target_mean)
                                        - _r2(action_pred, target, target_mean)
                                    ),
                                }
                            )

    summary: dict[str, Any] = {}
    metric_names = (
        "action_r2",
        "state_r2",
        "joint_r2",
        "state_incremental_r2_over_action",
    )
    for target_name in TARGET_NAMES:
        for basis_name in bases:
            for horizon in HORIZONS:
                for rank in RANKS:
                    rows = [
                        row
                        for row in records
                        if row["target"] == target_name
                        and row["basis"] == basis_name
                        and row["horizon"] == horizon
                        and row["rank"] == rank
                    ]
                    groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
                    value = {"windows": len(rows)}
                    for offset, metric in enumerate(metric_names):
                        value[metric] = _bootstrap_clustered(
                            np.asarray([row[metric] for row in rows]),
                            groups,
                            bootstrap_repeats,
                            20261040 + offset * 10000 + horizon * 100 + rank,
                        )
                    summary[f"{target_name}/{basis_name}/h{horizon}/k{rank}"] = value
    return {"records": records, "summary": summary}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]
    lines = [
        "# Rank- and target-specificity audit of frozen LeWM latent bases",
        "",
        "All probes are leave-one-episode-out. The action baseline uses observed future-action summaries; the reported joint-minus-action quantity is incremental predictive value, not a causal action effect.",
        "",
        "## Encoded-latent displacement: incremental R² over action",
        "",
        "| basis | h=1 k=1/2/4/8/16/32 | h=4 k=1/2/4/8/16/32 | h=8 k=1/2/4/8/16/32 |",
        "| --- | --- | --- | --- |",
    ]
    for basis in report["protocol"]["bases"]:
        cells = []
        for horizon in HORIZONS:
            values = [
                summary[f"dynamics_band_increment/{basis}/h{horizon}/k{rank}"]["state_incremental_r2_over_action"]["mean"]
                for rank in RANKS
            ]
            cells.append(" / ".join(f"{value:+.4f}" for value in values))
        lines.append(f"| {basis} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Residual-band latent displacement: incremental R² over action",
            "",
            "| basis | h=1 k=1/2/4/8/16/32 | h=4 k=1/2/4/8/16/32 | h=8 k=1/2/4/8/16/32 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for basis in report["protocol"]["bases"]:
        cells = []
        for horizon in HORIZONS:
            values = [
                summary[f"residual_band_increment/{basis}/h{horizon}/k{rank}"]["state_incremental_r2_over_action"]["mean"]
                for rank in RANKS
            ]
            cells.append(" / ".join(f"{value:+.4f}" for value in values))
        lines.append(f"| {basis} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Physical displacement: incremental R² over action",
            "",
            "| basis | h=1 k=1/2/4/8/16/32 | h=4 k=1/2/4/8/16/32 | h=8 k=1/2/4/8/16/32 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for basis in report["protocol"]["bases"]:
        cells = []
        for horizon in HORIZONS:
            values = [
                summary[f"physical_displacement/{basis}/h{horizon}/k{rank}"]["state_incremental_r2_over_action"]["mean"]
                for rank in RANKS
            ]
            cells.append(" / ".join(f"{value:+.4f}" for value in values))
        lines.append(f"| {basis} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The rank curve tests concentration and target specificity of source-fitted frozen subspaces; it does not select a rank after seeing a downstream task result.",
            "- Positive latent-target incremental R² supports encoded-latent predictive information. It does not imply physical-motion predictability.",
            "- Physical-target incremental R² uses observed actions and is therefore a conditional predictive diagnostic, not causal control or a physical counterfactual.",
            "- No online planning, task success, retraining gain, or universal world-model claim is made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    dynamics_basis_path: Path,
    residual_basis_path: Path,
    output: Path,
    policy: str = "pusht/lewm",
    device_name: str = "cpu",
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
    trajectories = _load_trajectories(
        model, val_root, device, normalization_mode, windows_per_episode, max_horizon
    )
    states = torch.load(val_root / "states.pth", map_location="cpu", weights_only=False).float()
    trajectories = _state_targets(trajectories, states, max_horizon)
    mean, bases = _load_all_bases(dynamics_basis_path, residual_basis_path, random_seed)
    crossfit = _crossfit(
        trajectories,
        actions,
        mean,
        bases,
        normalization_mode,
        action_mean,
        action_std,
        ridge,
        bootstrap_repeats,
    )
    report = {
        "artifact": "external_frozen_lewm_rank_target_specificity_audit",
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
            "state_fields": list(STATE_NAMES),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": list(HORIZONS),
            "ranks": list(RANKS),
            "bases": list(bases),
            "targets": list(TARGET_NAMES),
            "crossfit": "leave-one-episode-out",
            "action_features": "mean action sequence plus first and last block means",
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": action_source,
            },
            "ridge": float(ridge),
            "bootstrap": "episode-cluster bootstrap",
            "bootstrap_repeats": int(bootstrap_repeats),
            "random_basis_seed": int(random_seed),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "The source bases are fit on a separate frozen latent export and never refit on external validation windows.",
            "The rank curve is a frozen predictive representation diagnostic for encoded-latent and physical targets.",
            "The physical target uses observed action summaries; incremental R2 is not a causal action effect or physical counterfactual.",
            "The experiment does not establish online planning, task success, retraining improvement, or a universal world model.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("rank_target_specificity_notes.txt").write_text(
        "Rank/target specificity audit: source-fitted frozen bases, external leave-one-episode-out probes, no causal-control interpretation.\n"
    )
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
