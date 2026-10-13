"""Held-out physical-state observability audit for frozen external LeWM.

The public PushT validation archive contains five-dimensional simulator state
vectors.  This audit asks whether a linear probe fitted on true latent states
from other episodes can decode those variables from (i) true encoded future
embeddings and (ii) the frozen model's autoregressive embeddings.  It is a
physical observability diagnostic, not a decoder, environment intervention,
control, or planning evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _bootstrap_clustered,
    _load_trajectories,
    _normalization_parameters,
)


STATE_NAMES = ("agent_x", "agent_y", "object_x", "object_y", "object_angle")


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


def _fit_probe(source: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("probe source and target must have aligned two-dimensional rows")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered = source - source_mean
    covariance = centered.T @ centered / max(source.shape[0] - 1, 1)
    cross = centered.T @ (target - target_mean) / max(source.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / source.shape[1], 1e-8)
    weights = np.linalg.solve(
        covariance + float(ridge) * scale * np.eye(source.shape[1]), cross
    )
    return {"source_mean": source_mean, "target_mean": target_mean, "weights": weights}


def _decode(probe: dict[str, np.ndarray], source: np.ndarray) -> np.ndarray:
    return (source - probe["source_mean"]) @ probe["weights"] + probe["target_mean"]


def _normalized_errors(
    prediction: np.ndarray, target: np.ndarray, scale: np.ndarray
) -> tuple[float, np.ndarray]:
    normalized = (prediction - target) / scale
    # The caller passes one five-dimensional state at a time.  Preserve the
    # per-field absolute normalized error here; averaging across windows is
    # done only after the fold records have been assembled.
    return float(np.square(normalized).mean()), np.abs(normalized)


def _state_targets(
    trajectories: list[dict[str, Any]], states: torch.Tensor, max_horizon: int
) -> list[dict[str, Any]]:
    state_array = states.detach().cpu().numpy().astype(np.float64)
    output = []
    for row in trajectories:
        episode = int(row["episode_id"])
        start = int(row["start_observation"])
        indices = [FRAME_STRIDE * (start + step) for step in range(CONTEXT_LENGTH + max_horizon)]
        if max(indices) >= state_array.shape[1]:
            raise ValueError("state index exceeds public validation state sequence")
        # The first context_length observations are context; state t=0 is the
        # last context state, matching the latent trajectory convention.
        physical = state_array[episode, indices[CONTEXT_LENGTH - 1 :]]
        if physical.shape != row["true"].shape[:1] + (state_array.shape[-1],):
            raise ValueError("physical and latent trajectory lengths do not match")
        output.append({**row, "physical": physical})
    return output


def _crossfit(
    trajectories: list[dict[str, Any]],
    ridge: float,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    episode_ids = np.asarray(sorted({row["episode_id"] for row in trajectories}), dtype=np.int64)
    all_targets = np.concatenate([row["physical"] for row in trajectories], axis=0)
    scale = np.std(all_targets, axis=0, ddof=1)
    scale = np.maximum(scale, 1e-6)
    records: list[dict[str, Any]] = []
    for test_episode in episode_ids.tolist():
        train_rows = [row for row in trajectories if row["episode_id"] != test_episode]
        test_rows = [row for row in trajectories if row["episode_id"] == test_episode]
        train_latent = np.concatenate([row["true"] for row in train_rows], axis=0)
        train_target = np.concatenate([row["physical"] for row in train_rows], axis=0)
        probe = _fit_probe(train_latent, train_target, ridge=ridge)
        for row in test_rows:
            true_decoded = _decode(probe, row["true"])
            predicted_decoded = _decode(probe, row["predicted"])
            for horizon in range(row["physical"].shape[0]):
                true_mse, true_rmse = _normalized_errors(
                    true_decoded[horizon], row["physical"][horizon], scale
                )
                predicted_mse, predicted_rmse = _normalized_errors(
                    predicted_decoded[horizon], row["physical"][horizon], scale
                )
                records.append(
                    {
                        "episode_id": int(test_episode),
                        "start_observation": int(row["start_observation"]),
                        "horizon": int(horizon),
                        "true_decode_mse": true_mse,
                        "predicted_decode_mse": predicted_mse,
                        "predicted_minus_true_mse": predicted_mse - true_mse,
                        "true_decode_rmse_by_state": true_rmse.tolist(),
                        "predicted_decode_rmse_by_state": predicted_rmse.tolist(),
                        "predicted_minus_true_rmse_by_state": (predicted_rmse - true_rmse).tolist(),
                    }
                )

    summary: dict[str, Any] = {"by_horizon": {}}
    groups_all = np.asarray([row["episode_id"] for row in records], dtype=np.int64)
    for horizon in range(9):
        rows = [row for row in records if row["horizon"] == horizon]
        groups = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
        true_values = np.asarray([row["true_decode_mse"] for row in rows])
        predicted_values = np.asarray([row["predicted_decode_mse"] for row in rows])
        delta = np.asarray([row["predicted_minus_true_mse"] for row in rows])
        summary["by_horizon"][str(horizon)] = {
            "windows": int(len(rows)),
            "true_decode_mse": _bootstrap_clustered(true_values, groups, bootstrap_repeats, 20264000 + horizon),
            "predicted_decode_mse": _bootstrap_clustered(predicted_values, groups, bootstrap_repeats, 20265000 + horizon),
            "predicted_minus_true_mse": _bootstrap_clustered(delta, groups, bootstrap_repeats, 20266000 + horizon),
            "predicted_minus_true_rmse_by_state": [
                _bootstrap_clustered(
                    np.asarray(
                        [row["predicted_minus_true_rmse_by_state"][state_index] for row in rows]
                    ),
                    groups,
                    bootstrap_repeats,
                    20267000 + horizon * 10 + state_index,
                )
                for state_index in range(len(STATE_NAMES))
            ],
            "true_decode_rmse_by_state": np.mean(
                np.asarray([row["true_decode_rmse_by_state"] for row in rows]), axis=0
            ).tolist(),
            "predicted_decode_rmse_by_state": np.mean(
                np.asarray([row["predicted_decode_rmse_by_state"] for row in rows]), axis=0
            ).tolist(),
        }
    # The h=0 row is the held-out physical observability baseline.  Relative
    # changes at later horizons compare the same probe and state scale.
    baseline = summary["by_horizon"]["0"]["predicted_decode_mse"]["mean"]
    for horizon, value in summary["by_horizon"].items():
        value["predicted_mse_over_h0"] = float(
            value["predicted_decode_mse"]["mean"] / max(baseline, 1e-12)
        )
    return {
        "records": records,
        "summary": summary,
        "state_scale": scale.tolist(),
        "state_names": list(STATE_NAMES),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]["by_horizon"]
    lines = [
        "# External frozen LeWM physical-state observability audit",
        "",
        "A linear probe is fitted on true latent states from the other 20 episodes and evaluated on the held-out episode. Errors are normalized per physical state dimension by the external validation standard deviation.",
        "",
        "| h | true-latent decode MSE | predicted-latent decode MSE | predicted − true | predicted MSE / h=0 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon, value in summary.items():
        true_value = value["true_decode_mse"]
        predicted_value = value["predicted_decode_mse"]
        delta = value["predicted_minus_true_mse"]
        lines.append(
            f"| {horizon} | {true_value['mean']:.6g} [{true_value['ci95'][0]:.6g}, {true_value['ci95'][1]:.6g}] | "
            f"{predicted_value['mean']:.6g} [{predicted_value['ci95'][0]:.6g}, {predicted_value['ci95'][1]:.6g}] | "
            f"{delta['mean']:.6g} [{delta['ci95'][0]:.6g}, {delta['ci95'][1]:.6g}] | "
            f"{value['predicted_mse_over_h0']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Mean normalized RMSE by state at h=8 (agent_x, agent_y, object_x, object_y, object_angle):",
            "",
            "predicted minus true RMSE paired bootstrap intervals:",
            "",
            "| state | delta RMSE |",
            "| --- | ---: |",
            *[
                f"| {name} | {value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}] |"
                for name, value in zip(
                    STATE_NAMES, summary["8"]["predicted_minus_true_rmse_by_state"]
                )
            ],
            "",
            "## Boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    policy: str,
    output: Path,
    device_name: str = "cpu",
    normalization_mode: str = "public",
    windows_per_episode: int = 2,
    max_horizon: int = 8,
    ridge: float = 1e-2,
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, action_source = _normalization_parameters(actions, normalization_mode)
    trajectories = _load_trajectories(
        model,
        val_root,
        device,
        normalization_mode,
        windows_per_episode,
        max_horizon,
    )
    states = torch.load(val_root / "states.pth", map_location="cpu", weights_only=False).float()
    trajectories = _state_targets(trajectories, states, max_horizon)
    crossfit = _crossfit(trajectories, ridge=ridge, bootstrap_repeats=bootstrap_repeats)
    report = {
        "artifact": "external_frozen_lewm_physical_observability_audit",
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
            "frame_stride": FRAME_STRIDE,
            "max_horizon": int(max_horizon),
            "windows_per_episode": int(windows_per_episode),
            "probe": "ridge linear physical-state probe fitted on true latent states from non-held-out episodes",
            "crossfit": "leave-one-episode-out",
            "state_indexing": "states at the same frame indices as the encoded image windows; h=0 is the last context state",
            "normalization": "per-state-dimension standard deviation from all external validation windows",
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
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This measures physical-state observability of frozen latent embeddings on an external validation split.",
            "The probe is not a learned decoder used by the policy and does not establish physical counterfactual prediction, causal controllability, online planning, or task success.",
            "Predicted-minus-true decode error separates representation observability from the frozen model's latent rollout error; it is not a causal decomposition.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "External physical observability uses held-out episode probes and does not establish control.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    report = run_report(
        val_root=args.val_root,
        archive=args.archive,
        policy=args.policy,
        output=args.output,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        windows_per_episode=args.windows_per_episode,
        max_horizon=args.max_horizon,
        ridge=args.ridge,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(f"wrote {args.output} ({report['source']['windows']} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
