"""Cross-horizon readout transfer for frozen LeWorldModel latent subspaces.

This audit asks a stricter question than a within-horizon readout: a ridge
probe is fitted on one future horizon and evaluated on a different future
horizon on a held-out episode.  Source-fitted bases are fixed throughout.
The primary quantity is the joint-minus-action incremental R2, paired within
each held-out episode.  This is a frozen predictive diagnostic; it is not a
causal action effect, physical counterfactual, online planning result, or
task-success evaluation.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_features(
    rows: list[dict[str, Any]], mean: np.ndarray, basis: np.ndarray, rank: int
) -> np.ndarray:
    values = [
        (np.asarray(row["true"][0], dtype=np.float64) - mean) @ basis[:, :rank]
        for row in rows
    ]
    return np.stack(values)


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
        for source_horizon in HORIZONS:
            train_action = np.stack(
                [action_cache[(id(row), source_horizon)] for row in train_rows]
            )
            for target_horizon in HORIZONS:
                test_action = np.stack(
                    [action_cache[(id(row), target_horizon)] for row in test_rows]
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
                    # The source-horizon mean is fixed before the held-out
                    # episode is scored.  This keeps the cross-horizon R2
                    # comparable to the existing leakage-controlled audit.
                    target_mean = train_target.mean(axis=0)
                    action_probe = _fit_probe(train_action, train_target, ridge)
                    action_prediction = _decode(action_probe, test_action)
                    action_values = [
                        _r2(prediction, target, target_mean)
                        for prediction, target in zip(action_prediction, test_target)
                    ]
                    for basis_name, basis in bases.items():
                        for rank in RANKS:
                            train_state = _state_features(
                                train_rows, mean, basis, rank
                            )
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
                                state_r2 = _r2(state_pred, target, target_mean)
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
                                        "action_r2": float(action_r2),
                                        "state_r2": float(state_r2),
                                        "joint_r2": float(joint_r2),
                                        "state_incremental_r2_over_action": float(
                                            joint_r2 - action_r2
                                        ),
                                    }
                                )

    summary: dict[str, Any] = {}
    metrics = (
        "action_r2",
        "state_r2",
        "joint_r2",
        "state_incremental_r2_over_action",
    )
    for source_horizon in HORIZONS:
        for target_horizon in HORIZONS:
            for target_name in TARGET_NAMES:
                for basis_name in bases:
                    for rank in RANKS:
                        rows = [
                            row
                            for row in records
                            if row["source_horizon"] == source_horizon
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
                                20261050
                                + offset * 100000
                                + source_horizon * 10000
                                + target_horizon * 1000
                                + rank,
                            )
                        summary[
                            f"h{source_horizon}_to_h{target_horizon}/"
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
        ): row
        for row in records
    }
    contrasts: dict[str, Any] = {}
    contrast_pairs = (
        ("dynamics_basis", "covariance_basis", "dynamics_minus_covariance"),
        ("dynamics_basis", "random_basis", "dynamics_minus_random"),
        ("residual_basis", "dynamics_basis", "residual_minus_dynamics"),
    )
    for source_horizon in HORIZONS:
        for target_horizon in HORIZONS:
            for target_name in TARGET_NAMES:
                for rank in RANKS:
                    for left, right, contrast_name in contrast_pairs:
                        paired = []
                        groups = []
                        for row in records:
                            if (
                                row["source_horizon"] != source_horizon
                                or row["target_horizon"] != target_horizon
                                or row["target"] != target_name
                                or row["basis"] != left
                                or row["rank"] != rank
                            ):
                                continue
                            right_row = indexed.get(
                                (
                                    row["episode_id"],
                                    row["start_observation"],
                                    source_horizon,
                                    target_horizon,
                                    target_name,
                                    right,
                                    rank,
                                )
                            )
                            if right_row is None:
                                raise ValueError("paired contrast row is missing")
                            paired.append(
                                row["state_incremental_r2_over_action"]
                                - right_row["state_incremental_r2_over_action"]
                            )
                            groups.append(row["episode_id"])
                        contrasts[
                            f"h{source_horizon}_to_h{target_horizon}/"
                            f"{target_name}/k{rank}/{contrast_name}"
                        ] = _bootstrap_clustered(
                            np.asarray(paired, dtype=np.float64),
                            np.asarray(groups, dtype=np.int64),
                            bootstrap_repeats,
                            20261150
                            + source_horizon * 10000
                            + target_horizon * 1000
                            + rank,
                        )
    return {
        "records": records,
        "summary": summary,
        "paired_contrasts": contrasts,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]
    lines = [
        "# Cross-horizon readout transfer of frozen LeWM latent subspaces",
        "",
        "Each readout is fit on one source horizon and evaluated on a different target horizon on a held-out episode. The reported gain is joint-minus-action incremental R2, not a causal action effect.",
        "",
        "## Dynamics-band target, top-8 input",
        "",
        "| train horizon -> test horizon | dynamics | covariance | residual | random |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source_horizon in HORIZONS:
        for target_horizon in HORIZONS:
            values = []
            for basis_name in (
                "dynamics_basis",
                "covariance_basis",
                "residual_basis",
                "random_basis",
            ):
                key = (
                    f"h{source_horizon}_to_h{target_horizon}/"
                    f"dynamics_band_increment/{basis_name}/k8"
                )
                values.append(
                    summary[key]["state_incremental_r2_over_action"]["mean"]
                )
            lines.append(
                f"| h{source_horizon} -> h{target_horizon} | "
                + " | ".join(f"{value:+.4f}" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Residual-band target, top-8 input",
            "",
            "| train horizon -> test horizon | dynamics | covariance | residual | random |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for source_horizon in HORIZONS:
        for target_horizon in HORIZONS:
            values = []
            for basis_name in (
                "dynamics_basis",
                "covariance_basis",
                "residual_basis",
                "random_basis",
            ):
                key = (
                    f"h{source_horizon}_to_h{target_horizon}/"
                    f"residual_band_increment/{basis_name}/k8"
                )
                values.append(
                    summary[key]["state_incremental_r2_over_action"]["mean"]
                )
            lines.append(
                f"| h{source_horizon} -> h{target_horizon} | "
                + " | ".join(f"{value:+.4f}" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Paired cross-horizon contrasts at top-8",
            "",
            "Entries are paired incremental-R2 differences with 95% episode-cluster bootstrap intervals.",
            "",
            "| train horizon -> test horizon | dynamics minus covariance | dynamics minus random | residual minus dynamics |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source_horizon in HORIZONS:
        for target_horizon in HORIZONS:
            values = []
            for contrast_name in (
                "dynamics_minus_covariance",
                "dynamics_minus_random",
                "residual_minus_dynamics",
            ):
                key = (
                    f"h{source_horizon}_to_h{target_horizon}/"
                    f"dynamics_band_increment/k8/{contrast_name}"
                )
                result = report["crossfit"]["paired_contrasts"][key]
                values.append(
                    f"{result['mean']:+.4f} "
                    f"[{result['ci95'][0]:+.4f},{result['ci95'][1]:+.4f}]"
                )
            lines.append(
                f"| h{source_horizon} -> h{target_horizon} | "
                + " | ".join(values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A positive off-diagonal cell means that a current frozen subspace readout fitted at one temporal horizon retains incremental predictive value at another horizon.",
            "- Source bases are fitted on a separate frozen latent export; no basis is refit on the external validation episodes.",
            "- The physical target is included as a negative control and does not establish physical or causal dynamics.",
            "- The audit does not establish online planning, task success, retraining improvement, or a universal world model.",
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
    action_mean, action_std, action_source = _normalization_parameters(
        actions, normalization_mode
    )
    trajectories = _load_trajectories(
        model, val_root, device, normalization_mode, windows_per_episode, max_horizon
    )
    states = torch.load(val_root / "states.pth", map_location="cpu", weights_only=False).float()
    trajectories = _state_targets(trajectories, states, max_horizon)
    mean, bases = _load_all_bases(
        dynamics_basis_path, residual_basis_path, random_seed
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
        "artifact": "external_frozen_lewm_cross_horizon_readout",
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
            "crossfit": "leave-one-episode-out",
            "transfer": "fit at source horizon, evaluate at target horizon",
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
            "Off-diagonal cells are cross-horizon frozen predictive transfer diagnostics.",
            "Incremental R2 is predictive association beyond the declared action summary, not a causal action effect.",
            "The experiment does not establish physical causality, online planning, task success, retraining improvement, or a universal world model.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("cross_horizon_readout_notes.txt").write_text(
        "Cross-horizon readout: fit at one horizon, evaluate at another; frozen source bases; no causal-control interpretation.\n"
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
    parser.add_argument(
        "--action-normalization",
        choices=["public", "val_zscore", "raw_div100"],
        default="public",
    )
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
