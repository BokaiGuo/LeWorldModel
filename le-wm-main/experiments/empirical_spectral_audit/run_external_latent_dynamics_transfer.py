"""Cross-source multi-horizon latent-transition transfer audit.

This audit asks a narrower question than physical-state decoding: whether a
dynamics-sensitive basis extracted from a separate frozen latent export
transfers to future *latent increments* on held-out public PushT episodes.
The basis is never refit on the external windows.  For every leave-one-
episode-out fold, ridge probes compare action-only, current-state-only, and
joint action/state features.  The same probe is evaluated on both true encoded
trajectories and the frozen checkpoint's autoregressive trajectories.

The result is a frozen predictive diagnostic.  It is not a physical
counterfactual, causal-control, online-planning, task-success, or retraining
result.
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
    BANDS,
    _band_coordinates,
    _random_basis,
)
from experiments.empirical_spectral_audit.run_external_physical_dynamics_probe import (  # noqa: E402
    _action_summary,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    _bootstrap_clustered,
    _load_trajectories,
    _normalization_parameters,
)


HORIZONS = (1, 4, 8)
FEATURES = (
    "action_only",
    "state_only",
    "action_plus_state",
    "rollout_increment_only",
    "action_plus_rollout_increment",
)


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


def _load_transfer_bases(path: Path, random_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    required = {"latent_mean", "covariance_basis", "dynamics_basis", "conditional_dynamics_basis"}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"basis artifact missing fields: {missing}")
        mean = np.asarray(archive["latent_mean"], dtype=np.float64)
        bases = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in ("covariance_basis", "dynamics_basis", "conditional_dynamics_basis")
        }
    dim = int(mean.shape[0])
    bases["random_basis"] = _random_basis(dim, random_seed)
    for name, basis in bases.items():
        if basis.shape != (dim, dim) or not np.isfinite(basis).all():
            raise ValueError(f"invalid basis shape or values: {name}")
        if not np.allclose(basis.T @ basis, np.eye(dim), atol=1e-5):
            raise ValueError(f"basis is not orthonormal: {name}")
    return mean, bases


def _fit_probe(source: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("probe source and target must have aligned two-dimensional rows")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_source.T @ centered_source / max(source.shape[0] - 1, 1)
    cross = centered_source.T @ centered_target / max(source.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / max(source.shape[1], 1), 1e-8)
    weights = np.linalg.solve(
        covariance + float(ridge) * scale * np.eye(source.shape[1]), cross
    )
    return {"source_mean": source_mean, "target_mean": target_mean, "weights": weights}


def _decode(probe: dict[str, np.ndarray], source: np.ndarray) -> np.ndarray:
    return (np.asarray(source) - probe["source_mean"]) @ probe["weights"] + probe["target_mean"]


def _r2(prediction: np.ndarray, target: np.ndarray, target_mean: np.ndarray) -> float:
    numerator = float(np.square(np.asarray(prediction) - target).mean())
    denominator = max(float(np.square(target - target_mean).mean()), 1e-12)
    return float(1.0 - numerator / denominator)


def _feature_matrix(
    rows: list[dict[str, Any]],
    source_name: str,
    basis: np.ndarray,
    band: tuple[int, int],
    mean: np.ndarray,
    actions: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    horizon: int,
    normalization_mode: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    state_features = np.stack(
        [_band_coordinates(row[source_name], mean, basis, band)[0] for row in rows]
    )
    increment_features = np.stack(
        [
            _band_coordinates(row[source_name], mean, basis, band)[horizon]
            - _band_coordinates(row[source_name], mean, basis, band)[0]
            for row in rows
        ]
    )
    action_features = np.stack(
        [
            _action_summary(
                actions,
                int(row["episode_id"]),
                int(row["start_observation"]),
                horizon,
                normalization_mode,
                action_mean,
                action_std,
            )
            for row in rows
        ]
    )
    return {
        "action_only": action_features,
        "state_only": state_features,
        "action_plus_state": np.concatenate([action_features, state_features], axis=1),
        "rollout_increment_only": increment_features,
        "action_plus_rollout_increment": np.concatenate(
            [action_features, increment_features], axis=1
        ),
    }, state_features


def _target_matrix(
    rows: list[dict[str, Any]],
    source_name: str,
    basis: np.ndarray,
    band: tuple[int, int],
    mean: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return np.stack(
        [
            (
                _band_coordinates(row[source_name], mean, basis, band)[horizon]
                - _band_coordinates(row[source_name], mean, basis, band)[0]
            )
            for row in rows
        ]
    )


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
    records: list[dict[str, Any]] = []
    for test_episode in episode_ids:
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        for basis_name, basis in bases.items():
            for band_name, band in BANDS.items():
                for horizon in HORIZONS:
                    train_features, _ = _feature_matrix(
                        train_rows, "true", basis, band, mean, actions, action_mean, action_std,
                        horizon, normalization_mode,
                    )
                    train_target = _target_matrix(
                        train_rows, "true", basis, band, mean, horizon
                    )
                    target_mean = train_target.mean(axis=0)
                    probes = {
                        name: _fit_probe(features, train_target, ridge)
                        for name, features in train_features.items()
                    }
                    test_features, _ = _feature_matrix(
                        test_rows, "true", basis, band, mean, actions, action_mean, action_std,
                        horizon, normalization_mode,
                    )
                    true_target = _target_matrix(
                        test_rows, "true", basis, band, mean, horizon
                    )
                    predicted_features, _ = _feature_matrix(
                        test_rows, "predicted", basis, band, mean, actions, action_mean, action_std,
                        horizon, normalization_mode,
                    )
                    for feature_name, probe in probes.items():
                        true_prediction = _decode(probe, test_features[feature_name])
                        true_r2 = _r2(true_prediction, true_target, target_mean)
                        # The probe is trained only on true encoded transitions;
                        # applying it to autoregressive feature values isolates
                        # transport of the same latent transition signal.
                        predicted_prediction = _decode(probe, predicted_features[feature_name])
                        predicted_r2 = _r2(predicted_prediction, true_target, target_mean)
                        for row_index, row in enumerate(test_rows):
                            records.append(
                                {
                                    "episode_id": int(test_episode),
                                    "start_observation": int(row["start_observation"]),
                                    "basis": basis_name,
                                    "band": band_name,
                                    "horizon": int(horizon),
                                    "feature": feature_name,
                                    "true_r2": true_r2,
                                    "predicted_r2": predicted_r2,
                                    "true_sample_r2": _r2(
                                        true_prediction[row_index:row_index + 1],
                                        true_target[row_index:row_index + 1],
                                        target_mean,
                                    ),
                                    "predicted_sample_r2": _r2(
                                        predicted_prediction[row_index:row_index + 1],
                                        true_target[row_index:row_index + 1],
                                        target_mean,
                                    ),
                                }
                            )

    summary: dict[str, Any] = {"by_basis_band_horizon_feature": {}}
    for basis_name in bases:
        for band_name in BANDS:
            for horizon in HORIZONS:
                for feature_name in FEATURES:
                    key = f"{basis_name}/{band_name}/h{horizon}/{feature_name}"
                    rows = [
                        row for row in records
                        if row["basis"] == basis_name
                        and row["band"] == band_name
                        and row["horizon"] == horizon
                        and row["feature"] == feature_name
                    ]
                    groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
                    true_values = np.asarray([row["true_sample_r2"] for row in rows])
                    predicted_values = np.asarray([row["predicted_sample_r2"] for row in rows])
                    summary["by_basis_band_horizon_feature"][key] = {
                        "windows": len(rows),
                        "true_r2": _bootstrap_clustered(
                            true_values, groups, bootstrap_repeats, 20267000 + horizon
                        ),
                        "predicted_r2": _bootstrap_clustered(
                            predicted_values, groups, bootstrap_repeats, 20268000 + horizon
                        ),
                    }
    # Direct paired contrasts are computed from the same held-out windows and
    # are therefore more informative than comparing separately bootstrapped
    # intervals.
    contrasts: dict[str, Any] = {}
    for basis_name in bases:
        for band_name in BANDS:
            for horizon in HORIZONS:
                for source in ("true", "predicted"):
                    for feature_name in ("state_only", "action_plus_state"):
                        rows_joint = [
                            row for row in records
                            if row["basis"] == basis_name and row["band"] == band_name
                            and row["horizon"] == horizon and row["feature"] == feature_name
                        ]
                        rows_action = [
                            row for row in records
                            if row["basis"] == basis_name and row["band"] == band_name
                            and row["horizon"] == horizon and row["feature"] == "action_only"
                        ]
                        metric = f"{source}_sample_r2"
                        deltas = np.asarray([
                            left[metric] - right[metric]
                            for left, right in zip(rows_joint, rows_action)
                        ])
                        groups = np.asarray([row["episode_id"] for row in rows_joint], dtype=np.int64)
                        contrasts[
                            f"{basis_name}/{band_name}/h{horizon}/{source}/{feature_name}-minus-action"
                        ] = _bootstrap_clustered(
                            deltas, groups, bootstrap_repeats, 20269000 + horizon
                        )
    transport_gaps: dict[str, Any] = {}
    for basis_name in bases:
        for band_name in BANDS:
            for horizon in HORIZONS:
                for feature_name in FEATURES:
                    rows = [
                        row for row in records
                        if row["basis"] == basis_name
                        and row["band"] == band_name
                        and row["horizon"] == horizon
                        and row["feature"] == feature_name
                    ]
                    groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
                    gaps = np.asarray([
                        row["predicted_sample_r2"] - row["true_sample_r2"]
                        for row in rows
                    ])
                    transport_gaps[
                        f"{basis_name}/{band_name}/h{horizon}/{feature_name}"
                    ] = _bootstrap_clustered(
                        gaps, groups, bootstrap_repeats, 20269500 + horizon
                    )
    basis_contrasts: dict[str, Any] = {}
    focal_pairs = (
        ("dynamics_basis", "covariance_basis"),
        ("dynamics_basis", "random_basis"),
        ("conditional_dynamics_basis", "covariance_basis"),
    )
    for focal, control in focal_pairs:
        for band_name in BANDS:
            for horizon in HORIZONS:
                for feature_name in ("rollout_increment_only", "action_plus_rollout_increment"):
                    focal_rows = {
                        (row["episode_id"], row["start_observation"]): row
                        for row in records
                        if row["basis"] == focal
                        and row["band"] == band_name
                        and row["horizon"] == horizon
                        and row["feature"] == feature_name
                    }
                    control_rows = {
                        (row["episode_id"], row["start_observation"]): row
                        for row in records
                        if row["basis"] == control
                        and row["band"] == band_name
                        and row["horizon"] == horizon
                        and row["feature"] == feature_name
                    }
                    keys = sorted(set(focal_rows) & set(control_rows))
                    values = np.asarray([
                        focal_rows[key]["predicted_sample_r2"]
                        - control_rows[key]["predicted_sample_r2"]
                        for key in keys
                    ])
                    groups = np.asarray([key[0] for key in keys], dtype=np.int64)
                    basis_contrasts[
                        f"{focal}_minus_{control}/{band_name}/h{horizon}/{feature_name}"
                    ] = _bootstrap_clustered(
                        values, groups, bootstrap_repeats, 20269650 + horizon
                    )
    return {
        "records": records,
        "summary": summary,
        "contrasts": contrasts,
        "transport_gaps": transport_gaps,
        "basis_contrasts": basis_contrasts,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]["by_basis_band_horizon_feature"]
    contrasts = report["crossfit"]["contrasts"]
    lines = [
        "# External frozen latent-dynamics transfer audit",
        "",
        "The transferred basis is fit on a separate frozen latent export and is not refit on the public external windows. Probes are leave-one-episode-out. `true` evaluates features from encoded future trajectories; `predicted` applies the same true-trajectory probe to the frozen autoregressive trajectory features. Values are episode-cluster bootstrap means and 95% intervals over held-out windows.",
        "",
        "## Top-8 band",
        "",
        "| basis | horizon | feature | true latent-increment R2 | predicted latent-increment R2 | predicted minus true |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    fmt = lambda value: f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
    for basis_name in report["protocol"]["bases"]:
        for horizon in HORIZONS:
            for feature_name in FEATURES:
                key = f"{basis_name}/top8/h{horizon}/{feature_name}"
                value = summary[key]
                gap = report["crossfit"]["transport_gaps"][key]
                lines.append(
                    f"| {basis_name} | {horizon} | {feature_name} | {fmt(value['true_r2'])} | {fmt(value['predicted_r2'])} | {fmt(gap)} |"
                )
    lines.extend([
        "",
        "## Paired basis contrasts for rollout increments",
        "",
        "| contrast | band | horizon | feature | predicted R2 difference |",
        "| --- | --- | ---: | --- | ---: |",
    ])
    for contrast_name in (
        "dynamics_basis_minus_covariance_basis",
        "dynamics_basis_minus_random_basis",
        "conditional_dynamics_basis_minus_covariance_basis",
    ):
        for band_name in ("top8", "middle8_32"):
            for horizon in HORIZONS:
                key = f"{contrast_name}/{band_name}/h{horizon}/rollout_increment_only"
                lines.append(
                    f"| {contrast_name} | {band_name} | {horizon} | rollout_increment_only | {fmt(report['crossfit']['basis_contrasts'][key])} |"
                )
    lines.extend([
        "",
        "## Action-plus-state gain over action-only",
        "",
        "| basis | band | horizon | source | joint minus action R2 |",
        "| --- | --- | ---: | --- | ---: |",
    ])
    for basis_name in report["protocol"]["bases"]:
        for band_name in BANDS:
            for horizon in HORIZONS:
                for source in ("true", "predicted"):
                    key = f"{basis_name}/{band_name}/h{horizon}/{source}/action_plus_state-minus-action"
                    lines.append(
                        f"| {basis_name} | {band_name} | {horizon} | {source} | {fmt(contrasts[key])} |"
                    )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        *[f"- {item}" for item in report["claim_boundary"]],
    ])
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    basis_path: Path,
    output: Path,
    policy: str = "pusht/lewm",
    device_name: str = "cpu",
    normalization_mode: str = "public",
    windows_per_episode: int = 8,
    max_horizon: int = 8,
    random_seed: int = 20261003,
    ridge: float = 1e-2,
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    if max_horizon != 8:
        raise ValueError("the external protocol requires max_horizon=8")
    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, action_source = _normalization_parameters(actions, normalization_mode)
    trajectories = _load_trajectories(
        model, val_root, device, normalization_mode, windows_per_episode, max_horizon
    )
    mean, bases = _load_transfer_bases(basis_path, random_seed)
    # The transfer artifact intentionally contains only bases available from
    # the internal frozen export.  No basis is fit from external trajectories.
    crossfit = _crossfit(
        trajectories, actions, mean, bases, normalization_mode,
        action_mean, action_std, ridge, bootstrap_repeats,
    )
    report = {
        "artifact": "external_frozen_latent_dynamics_transfer_audit",
        "status": "complete",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "basis_artifact": str(basis_path),
            "basis_sha256": _sha256(basis_path),
            "episodes": int(len({row["episode_id"] for row in trajectories})),
            "windows": int(len(trajectories)),
            "latent_dim": int(mean.shape[0]),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": 3,
            "frame_stride": 5,
            "horizons": list(HORIZONS),
            "windows_per_episode": int(windows_per_episode),
            "bases": list(bases),
            "bands": BANDS,
            "target": "future latent increment in the transferred basis band",
            "features": list(FEATURES),
            "crossfit": "leave-one-episode-out over external episodes",
            "basis_fit": "separate internal frozen latent export; no external refitting",
            "probe_fit": "ridge on true encoded trajectories from training episodes",
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": action_source,
                "original_lewm_training_statistics_available": False,
            },
            "ridge": float(ridge),
            "bootstrap": "episode-cluster bootstrap",
            "bootstrap_repeats": int(bootstrap_repeats),
            "random_basis_seed": int(random_seed),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This transfers a basis learned from a separate frozen latent export to external held-out latent trajectories.",
            "Positive R2 is evidence of predictive information in the selected latent band, not evidence of a causal latent mechanism.",
            "The action-plus-state contrast is an observed-action predictive diagnostic and does not identify causal action effects.",
            "True-versus-predicted comparisons diagnose transport through the frozen autoregressive rollout; they do not establish physical counterfactual prediction.",
            "The result does not establish causal controllability, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "Cross-source multi-horizon latent-dynamics transfer audit completed.\n"
        "The basis is frozen from a separate internal export; external probes are episode-disjoint.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20261003)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        archive=args.archive,
        basis_path=args.basis,
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
