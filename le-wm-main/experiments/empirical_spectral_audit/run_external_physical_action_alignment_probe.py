"""Test action--physical-displacement alignment with permutation controls."""

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

from experiments.empirical_spectral_audit.run_external_physical_dynamics_probe import (  # noqa: E402
    HORIZONS,
    _action_summary,
    _finite,
    _r2,
    _physical_displacement,
)
from experiments.empirical_spectral_audit.run_external_physical_observability_audit import (  # noqa: E402
    STATE_NAMES,
    _fit_probe,
    _state_targets,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
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


def _action_features(
    trajectories: list[dict[str, Any]],
    actions: torch.Tensor,
    horizon: int,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> np.ndarray:
    return np.stack(
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
            for row in trajectories
        ]
    )


def _permutations(trajectories: list[dict[str, Any]], seed: int) -> dict[str, np.ndarray]:
    n = len(trajectories)
    by_episode: dict[int, list[int]] = {}
    for index, row in enumerate(trajectories):
        by_episode.setdefault(int(row["episode_id"]), []).append(index)
    within = np.arange(n, dtype=np.int64)
    for indices in by_episode.values():
        ordered = sorted(indices, key=lambda index: int(trajectories[index]["start_observation"]))
        within[ordered] = np.roll(np.asarray(ordered, dtype=np.int64), -1)
    # A cyclic shift is a deterministic derangement for n>1; unlike a raw
    # random permutation it cannot accidentally leave some action windows in
    # their original positions.
    global_derangement = np.roll(np.arange(n, dtype=np.int64), -1)
    return {
        "aligned": np.arange(n, dtype=np.int64),
        "within_episode_shift": within,
        "global_derangement": global_derangement,
    }


def _crossfit(
    trajectories: list[dict[str, Any]],
    actions: torch.Tensor,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    all_targets = {
        horizon: np.stack([_physical_displacement(row["physical"], horizon) for row in trajectories])
        for horizon in HORIZONS
    }
    scale = {
        horizon: np.maximum(np.std(target, axis=0, ddof=1), 1e-6)
        for horizon, target in all_targets.items()
    }
    permutations = _permutations(trajectories, seed)
    episode_ids = sorted({int(row["episode_id"]) for row in trajectories})
    records: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        features = _action_features(
            trajectories, actions, horizon, normalization_mode, action_mean, action_std
        )
        target = all_targets[horizon]
        for test_episode in episode_ids:
            train_mask = np.asarray(
                [int(row["episode_id"]) != test_episode for row in trajectories], dtype=bool
            )
            test_mask = ~train_mask
            train_target = target[train_mask]
            target_mean = train_target.mean(axis=0)
            for condition, permutation in permutations.items():
                permuted_features = features[permutation]
                probe = _fit_probe(permuted_features[train_mask], train_target, ridge=1e-2)
                decoded = (permuted_features[test_mask] - probe["source_mean"]) @ probe[
                    "weights"
                ] + probe["target_mean"]
                normalized_mse = np.square(
                    (decoded - target[test_mask]) / scale[horizon]
                ).mean(axis=1)
                r2 = np.asarray(
                    [
                        _r2(prediction, truth, target_mean)
                        for prediction, truth in zip(decoded, target[test_mask])
                    ],
                    dtype=np.float64,
                )
                for row, mse, row_r2 in zip(
                    np.asarray([row for row in trajectories if int(row["episode_id"]) == test_episode], dtype=object),
                    normalized_mse,
                    r2,
                ):
                    records.append(
                        {
                            "episode_id": int(test_episode),
                            "start_observation": int(row["start_observation"]),
                            "condition": condition,
                            "horizon": int(horizon),
                            "normalized_mse": float(mse),
                            "r2": float(row_r2),
                        }
                    )
    summary: dict[str, Any] = {"by_condition_horizon": {}}
    for condition in permutations:
        for horizon in HORIZONS:
            rows = [
                row
                for row in records
                if row["condition"] == condition and row["horizon"] == horizon
            ]
            groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
            summary["by_condition_horizon"][f"{condition}/h{horizon}"] = {
                "normalized_mse": _bootstrap_clustered(
                    np.asarray([row["normalized_mse"] for row in rows]),
                    groups,
                    bootstrap_repeats,
                    seed + 1000 + horizon,
                ),
                "r2": _bootstrap_clustered(
                    np.asarray([row["r2"] for row in rows]),
                    groups,
                    bootstrap_repeats,
                    seed + 2000 + horizon,
                ),
            }
    for horizon in HORIZONS:
        aligned = [
            row
            for row in records
            if row["condition"] == "aligned" and row["horizon"] == horizon
        ]
        aligned_by_key = {
            (row["episode_id"], row["start_observation"]): row for row in aligned
        }
        for condition in ("within_episode_shift", "global_derangement"):
            control = [
                row
                for row in records
                if row["condition"] == condition and row["horizon"] == horizon
            ]
            delta_mse = np.asarray(
                [
                    row["normalized_mse"]
                    - aligned_by_key[(row["episode_id"], row["start_observation"])]
                    ["normalized_mse"]
                    for row in control
                ]
            )
            delta_r2 = np.asarray(
                [
                    row["r2"]
                    - aligned_by_key[(row["episode_id"], row["start_observation"])]
                    ["r2"]
                    for row in control
                ]
            )
            groups = np.asarray([row["episode_id"] for row in control], dtype=np.int64)
            summary["by_condition_horizon"][f"{condition}/h{horizon}"]["delta_vs_aligned"] = {
                "normalized_mse": _bootstrap_clustered(
                    delta_mse, groups, bootstrap_repeats, seed + 3000 + horizon
                ),
                "r2": _bootstrap_clustered(
                    delta_r2, groups, bootstrap_repeats, seed + 4000 + horizon
                ),
            }
    return {"records": records, "summary": summary, "permutations": {key: value.tolist() for key, value in permutations.items()}}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]["by_condition_horizon"]
    lines = [
        "# External physical-displacement action-alignment probe",
        "",
        "The aligned condition uses the action blocks belonging to each transition. The within-episode shift and global derangement preserve action statistics while breaking action--transition alignment. All probes are leave-one-episode-out and use episode-cluster bootstrap.",
        "",
        "| condition | h | normalized MSE | R2 | delta normalized MSE vs aligned | delta R2 vs aligned |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fmt = lambda value: f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"
    for condition in ("aligned", "within_episode_shift", "global_derangement"):
        for horizon in HORIZONS:
            value = summary[f"{condition}/h{horizon}"]
            if condition == "aligned":
                delta_mse = "--"
                delta_r2 = "--"
            else:
                delta_mse = fmt(value["delta_vs_aligned"]["normalized_mse"])
                delta_r2 = fmt(value["delta_vs_aligned"]["r2"])
            lines.append(
                f"| {condition} | {horizon} | {fmt(value['normalized_mse'])} | {fmt(value['r2'])} | {delta_mse} | {delta_r2} |"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Positive derangement-minus-aligned error and R2 contrasts show that the observed action signal depends on transition alignment rather than only its marginal distribution.",
            "- This is an observed-action predictive control, not a physical counterfactual, causal-control, online-planning, or task-success evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    output: Path,
    policy: str = "pusht/lewm",
    device_name: str = "cpu",
    normalization_mode: str = "public",
    windows_per_episode: int = 2,
    max_horizon: int = 8,
    bootstrap_repeats: int = 2000,
    seed: int = 20261003,
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
    crossfit = _crossfit(
        trajectories,
        actions,
        normalization_mode,
        action_mean,
        action_std,
        bootstrap_repeats,
        seed,
    )
    report = {
        "artifact": "external_frozen_lewm_physical_action_alignment_probe",
        "status": "complete",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "episodes": int(len({row["episode_id"] for row in trajectories})),
            "windows": int(len(trajectories)),
            "state_fields": list(STATE_NAMES),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "max_horizon": int(max_horizon),
            "horizons": list(HORIZONS),
            "windows_per_episode": int(windows_per_episode),
            "probe": "ridge physical-displacement readout from observed action summaries",
            "crossfit": "leave-one-episode-out",
            "action_permutations": "within-episode cyclic shift and global derangement",
            "action_normalization": {
                "mode": normalization_mode,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": action_source,
            },
            "bootstrap": "episode-cluster bootstrap",
            "bootstrap_repeats": int(bootstrap_repeats),
            "seed": int(seed),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This tests whether observed action summaries predict aligned external physical displacement better than alignment-preserving permutation controls.",
            "It is not a physical counterfactual or causal action-effect estimate because no alternative environment trajectory is generated.",
            "It does not establish causal controllability, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    run_report(
        val_root=args.val_root,
        archive=args.archive,
        output=args.output,
        policy=args.policy,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        windows_per_episode=args.windows_per_episode,
        max_horizon=args.max_horizon,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
