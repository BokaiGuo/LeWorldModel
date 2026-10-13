"""Probe physical displacement from transferred frozen latent increments.

This audit is deliberately different from a static physical-state readout.
For each external PushT window it predicts the physical displacement from the
last context state to a future state using (i) a selected frozen latent-band
increment and (ii) an action-only summary.  Bases are fitted on a separate
frozen latent export and probes are fit/evaluated leave-one-episode-out.

The output is a frozen predictive diagnostic.  It is not a physical
counterfactual, causal-control, online-planning, or task-success result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    BASIS_NAMES,
    _band_coordinates,
    _load_bases,
)
from experiments.empirical_spectral_audit.run_external_physical_observability_audit import (  # noqa: E402
    STATE_NAMES,
    _decode,
    _fit_probe,
    _state_targets,
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


HORIZONS = (1, 4, 8)


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


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    result = np.asarray(delta, dtype=np.float64).copy()
    result[..., 4] = (result[..., 4] + np.pi) % (2.0 * np.pi) - np.pi
    return result


def _physical_displacement(physical: np.ndarray, horizon: int) -> np.ndarray:
    if physical.ndim != 2 or physical.shape[1] != len(STATE_NAMES):
        raise ValueError("physical trajectory must have shape (time, five state fields)")
    if not 0 <= horizon < physical.shape[0]:
        raise ValueError("horizon exceeds physical trajectory")
    return _wrap_angle(physical[horizon] - physical[0])


def _action_summary(
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
    blocks = []
    first_obs = int(start_observation) + CONTEXT_LENGTH - 1
    for step in range(int(horizon)):
        obs = first_obs + step
        block = normalized[obs * FRAME_STRIDE : (obs + 1) * FRAME_STRIDE]
        if block.shape[0] < FRAME_STRIDE:
            block = torch.cat(
                [block, torch.zeros((FRAME_STRIDE - block.shape[0], 2), dtype=block.dtype)]
            )
        blocks.append(block.mean(dim=0).numpy())
    sequence = np.asarray(blocks, dtype=np.float64)
    # A compact summary avoids making the h=8 action-only baseline
    # underdetermined with 40 training windows per fold.
    return np.concatenate([sequence.mean(axis=0), sequence[0], sequence[-1]])


def _r2(prediction: np.ndarray, target: np.ndarray, target_mean: np.ndarray) -> float:
    numerator = float(np.square(prediction - target).mean())
    denominator = max(float(np.square(target - target_mean).mean()), 1e-12)
    return float(1.0 - numerator / denominator)


def _crossfit(
    trajectories: list[dict[str, Any]],
    raw_actions: torch.Tensor,
    mean: np.ndarray,
    bases: dict[str, np.ndarray],
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    ridge: float,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    all_targets = np.concatenate(
        [_physical_displacement(row["physical"], h)[None] for row in trajectories for h in HORIZONS],
        axis=0,
    )
    scale = np.maximum(np.std(all_targets, axis=0, ddof=1), 1e-6)
    episode_ids = sorted({int(row["episode_id"]) for row in trajectories})
    records: list[dict[str, Any]] = []
    for test_episode in episode_ids:
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        for horizon in HORIZONS:
            train_target = np.stack([_physical_displacement(row["physical"], horizon) for row in train_rows])
            target_mean = train_target.mean(axis=0)
            train_actions = np.stack(
                [
                    _action_summary(
                        raw_actions,
                        int(row["episode_id"]),
                        int(row["start_observation"]),
                        horizon,
                        normalization_mode,
                        action_mean,
                        action_std,
                    )
                    for row in train_rows
                ]
            )
            action_probe = _fit_probe(train_actions, train_target, ridge=ridge)
            for basis_name, basis in bases.items():
                for band_name, band in BANDS.items():
                    train_increments = np.stack(
                        [
                            _band_coordinates(row["true"], mean, basis, band)[horizon]
                            - _band_coordinates(row["true"], mean, basis, band)[0]
                            for row in train_rows
                        ]
                    )
                    probe = _fit_probe(train_increments, train_target, ridge=ridge)
                    train_pairs = np.concatenate(
                        [
                            np.stack(
                                [
                                    _band_coordinates(row["true"], mean, basis, band)[0]
                                    for row in train_rows
                                ]
                            ),
                            np.stack(
                                [
                                    _band_coordinates(row["true"], mean, basis, band)[horizon]
                                    for row in train_rows
                                ]
                            ),
                        ],
                        axis=1,
                    )
                    pair_probe = _fit_probe(train_pairs, train_target, ridge=ridge)
                    augmented_probe = _fit_probe(
                        np.concatenate([train_actions, train_increments], axis=1),
                        train_target,
                        ridge=ridge,
                    )
                    for row in test_rows:
                        target = _physical_displacement(row["physical"], horizon)
                        action_features = _action_summary(
                            raw_actions,
                            int(row["episode_id"]),
                            int(row["start_observation"]),
                            horizon,
                            normalization_mode,
                            action_mean,
                            action_std,
                        )[None]
                        true_band = _band_coordinates(row["true"], mean, basis, band)
                        predicted_band = _band_coordinates(row["predicted"], mean, basis, band)
                        true_increment = (true_band[horizon] - true_band[0])[None]
                        predicted_increment = (predicted_band[horizon] - predicted_band[0])[None]
                        true_pair = np.concatenate([true_band[0], true_band[horizon]])[None]
                        predicted_pair = np.concatenate(
                            [predicted_band[0], predicted_band[horizon]]
                        )[None]
                        action_prediction = _decode(action_probe, action_features)[0]
                        true_prediction = _decode(probe, true_increment)[0]
                        predicted_prediction = _decode(probe, predicted_increment)[0]
                        true_pair_prediction = _decode(pair_probe, true_pair)[0]
                        predicted_pair_prediction = _decode(pair_probe, predicted_pair)[0]
                        true_augmented = _decode(
                            augmented_probe,
                            np.concatenate([action_features, true_increment], axis=1),
                        )[0]
                        predicted_augmented = _decode(
                            augmented_probe,
                            np.concatenate([action_features, predicted_increment], axis=1),
                        )[0]

                        def metrics(prediction: np.ndarray) -> tuple[float, float]:
                            normalized_mse = float(np.square((prediction - target) / scale).mean())
                            return normalized_mse, _r2(prediction, target, target_mean)

                        action_mse, action_r2 = metrics(action_prediction)
                        true_mse, true_r2 = metrics(true_prediction)
                        predicted_mse, predicted_r2 = metrics(predicted_prediction)
                        true_pair_mse, true_pair_r2 = metrics(true_pair_prediction)
                        predicted_pair_mse, predicted_pair_r2 = metrics(predicted_pair_prediction)
                        true_aug_mse, true_aug_r2 = metrics(true_augmented)
                        predicted_aug_mse, predicted_aug_r2 = metrics(predicted_augmented)
                        records.append(
                            {
                                "episode_id": int(test_episode),
                                "start_observation": int(row["start_observation"]),
                                "basis": basis_name,
                                "band": band_name,
                                "horizon": int(horizon),
                                "action_only_mse": action_mse,
                                "action_only_r2": action_r2,
                                "true_increment_mse": true_mse,
                                "true_increment_r2": true_r2,
                                "predicted_increment_mse": predicted_mse,
                                "predicted_increment_r2": predicted_r2,
                                "predicted_minus_true_mse": predicted_mse - true_mse,
                                "true_pair_mse": true_pair_mse,
                                "true_pair_r2": true_pair_r2,
                                "predicted_pair_mse": predicted_pair_mse,
                                "predicted_pair_r2": predicted_pair_r2,
                                "predicted_pair_minus_true_mse": predicted_pair_mse - true_pair_mse,
                                "true_action_plus_increment_r2": true_aug_r2,
                                "predicted_action_plus_increment_r2": predicted_aug_r2,
                                "true_incremental_r2": true_aug_r2 - action_r2,
                                "predicted_incremental_r2": predicted_aug_r2 - action_r2,
                                "true_mse_by_state": (
                                    np.square((true_prediction - target) / scale)
                                ).tolist(),
                                "predicted_mse_by_state": (
                                    np.square((predicted_prediction - target) / scale)
                                ).tolist(),
                            }
                        )

    summary: dict[str, Any] = {"by_basis_band_horizon": {}}
    metric_names = (
        "action_only_mse",
        "action_only_r2",
        "true_increment_mse",
        "true_increment_r2",
        "predicted_increment_mse",
        "predicted_increment_r2",
        "predicted_minus_true_mse",
        "true_pair_mse",
        "true_pair_r2",
        "predicted_pair_mse",
        "predicted_pair_r2",
        "predicted_pair_minus_true_mse",
        "true_action_plus_increment_r2",
        "predicted_action_plus_increment_r2",
        "true_incremental_r2",
        "predicted_incremental_r2",
    )
    for basis_name in bases:
        for band_name in BANDS:
            for horizon in HORIZONS:
                rows = [
                    row
                    for row in records
                    if row["basis"] == basis_name
                    and row["band"] == band_name
                    and row["horizon"] == horizon
                ]
                groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
                value = {"windows": len(rows)}
                for offset, metric_name in enumerate(metric_names):
                    value[metric_name] = _bootstrap_clustered(
                        np.asarray([row[metric_name] for row in rows]),
                        groups,
                        bootstrap_repeats,
                        20268000 + offset * 100 + horizon,
                    )
                summary["by_basis_band_horizon"][
                    f"{basis_name}/{band_name}/h{horizon}"
                ] = value
    return {"records": records, "summary": summary, "target_scale": scale.tolist()}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]["by_basis_band_horizon"]
    lines = [
        "# External physical-displacement probe from transferred latent increments",
        "",
        "The target is the five-dimensional physical displacement from the last context state to h=1, 4, or 8. Probes are fit on true latent trajectories from 20 episodes and evaluated on the held-out episode. Action-only is a compact observed-action summary; action-plus-increment reports the incremental predictive value of the selected latent band. The transition-pair columns use [z_t, z_{t+h}] and are a less restrictive test than using only z_{t+h}-z_t.",
        "",
        "## h=1",
        "",
        "| basis | band | true increment R2 | predicted increment R2 | true pair R2 | predicted pair R2 | predicted incremental R2 over action |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fmt = lambda value: f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
    for horizon in HORIZONS:
        if horizon != 1:
            lines.extend(
                [
                    "",
                    f"## h={horizon}",
                    "",
                    "| basis | band | true increment R2 | predicted increment R2 | true pair R2 | predicted pair R2 | predicted incremental R2 over action |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
        for basis_name in BASES_ORDER:
            for band_name in BANDS:
                value = summary[f"{basis_name}/{band_name}/h{horizon}"]
                lines.append(
                    f"| {basis_name} | {band_name} | {fmt(value['true_increment_r2'])} | "
                    f"{fmt(value['predicted_increment_r2'])} | {fmt(value['true_pair_r2'])} | "
                    f"{fmt(value['predicted_pair_r2'])} | {fmt(value['predicted_incremental_r2'])} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The latent bands and their bases are frozen and transferred from a separate latent export; all physical-displacement probes are leave-one-episode-out.",
            "- Absolute R2 is compared across bands only because the target is fixed; band dimensions still require dimension-matched random controls for any band-importance claim.",
            "- Action-plus-increment is an observed-action predictive diagnostic. It does not establish a causal action effect, physical counterfactual prediction, controllability, online planning, task success, or retraining improvement.",
            "- Positive true-latent R2 with weak predicted-latent R2 indicates that the physical motion is observable in encoded states but is not faithfully transported by the frozen autoregressive rollout.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


BASES_ORDER = list(BASIS_NAMES)


def run_report(
    val_root: Path,
    archive: Path,
    basis_path: Path,
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
    mean, bases = _load_bases(basis_path, random_seed)
    # The physical band probe has a fixed five-basis schema.  This audit also
    # accepts the optional action-residualized dynamics basis when present.
    with np.load(basis_path, allow_pickle=False) as archive_npz:
        if "conditional_dynamics_basis" in archive_npz.files:
            bases["conditional_dynamics_basis"] = np.asarray(
                archive_npz["conditional_dynamics_basis"], dtype=np.float64
            )
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
        "artifact": "external_frozen_lewm_physical_displacement_probe",
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
            "state_fields": list(STATE_NAMES),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": list(HORIZONS),
            "windows_per_episode": int(windows_per_episode),
            "bases": list(bases),
            "bands": BANDS,
            "target": "wrapped physical displacement from h=0 to h",
            "action_features": "mean action sequence plus first and last block means, six dimensions",
            "crossfit": "leave-one-episode-out",
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
            "This is a frozen external predictive diagnostic of physical displacement information in latent increments.",
            "The action-only and action-plus-increment models use observed future action summaries and therefore do not identify causal action effects.",
            "The transferred bases are not refit on the external validation windows; random-basis bands are dimension-matched controls.",
            "The result does not establish physical counterfactual prediction, causal controllability, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("physical_dynamics_probe_notes.txt").write_text(
        "Physical displacement probe: frozen transferred latent increments with action-only and random-basis controls. This is not causal control evidence.\n"
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
    parser.add_argument("--windows-per-episode", type=int, default=2)
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
