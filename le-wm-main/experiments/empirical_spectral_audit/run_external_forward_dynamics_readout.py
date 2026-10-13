"""Cross-fitted forward readout of external latent dynamics.

Unlike a projection-share audit, this script never uses the future latent
state as an input.  It predicts the full future latent increment from the
current latent coordinates and observed action summaries, leaving one whole
episode out at a time.  Source bases are fitted on a separate frozen export.

The result is a predictive representation diagnostic.  It is not a causal
action effect, physical counterfactual, online-control, planning, or task-
success result.
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
    _normalization_parameters,
)
from experiments.empirical_spectral_audit.run_external_trajectory_operator_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    _load_trajectories,
)
from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    _standardize_public_actions,
)


HORIZONS = (1, 4, 8)
BANDS = {"top8": (0, 8), "middle8_32": (8, 32)}
BASIS_NAMES = (
    "covariance_basis",
    "dynamics_basis",
    "residual_basis",
    "residual_covariance_basis",
    "random_basis",
)
FEATURES = (
    "action_future",
    "action_past",
    "action_deranged",
    "state_only",
    "state_plus_action",
    "state_plus_past",
    "state_plus_deranged",
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


def _random_basis(dimension: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _load_bases(path: Path, random_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"latent_mean", *BASIS_NAMES[:-1]}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"basis artifact missing {missing}")
        mean = np.asarray(archive["latent_mean"], dtype=np.float64)
        bases = {name: np.asarray(archive[name], dtype=np.float64) for name in BASIS_NAMES[:-1]}
    dimension = int(mean.shape[0])
    bases["random_basis"] = _random_basis(dimension, random_seed)
    for name, basis in bases.items():
        if basis.shape != (dimension, dimension) or not np.isfinite(basis).all():
            raise ValueError(f"invalid basis {name}: {basis.shape}")
        if not np.allclose(basis.T @ basis, np.eye(dimension), atol=1e-5):
            raise ValueError(f"basis is not orthonormal: {name}")
    return mean, bases


def _action_features(
    raw_actions: torch.Tensor,
    episode_id: int,
    start_observation: int,
    horizon: int,
    offset: int,
    normalization_mode: str,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> np.ndarray:
    normalized = _standardize_public_actions(
        raw_actions[episode_id], normalization_mode, action_mean, action_std
    )
    current_observation = int(start_observation) + CONTEXT_LENGTH - 1
    blocks: list[np.ndarray] = []
    for step in range(int(horizon)):
        observation = current_observation + int(offset) + step
        begin, end = observation * FRAME_STRIDE, (observation + 1) * FRAME_STRIDE
        if observation < 0 or begin >= normalized.shape[0]:
            block = torch.zeros((FRAME_STRIDE, 2), dtype=normalized.dtype)
        else:
            block = normalized[max(begin, 0) : min(end, normalized.shape[0])]
            if block.shape[0] < FRAME_STRIDE:
                block = torch.cat(
                    [block, torch.zeros((FRAME_STRIDE - block.shape[0], 2), dtype=block.dtype)]
                )
        blocks.append(block.mean(dim=0).numpy())
    sequence = np.asarray(blocks, dtype=np.float64)
    return np.concatenate([sequence.mean(axis=0), sequence[0], sequence[-1]])


def _fit_probe(source: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_mean, target - target_mean
    covariance = x.T @ x / max(source.shape[0] - 1, 1)
    cross = x.T @ y / max(source.shape[0] - 1, 1)
    scale = max(float(np.trace(covariance)) / max(source.shape[1], 1), 1e-8)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "weights": np.linalg.solve(covariance + float(ridge) * scale * np.eye(source.shape[1]), cross),
    }


def _decode(probe: dict[str, np.ndarray], source: np.ndarray) -> np.ndarray:
    return (source - probe["source_mean"]) @ probe["weights"] + probe["target_mean"]


def _bootstrap(values: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite one-dimensional values")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(int(repeats), values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "episodes": int(values.size),
        "bootstrap_repeats": int(repeats),
        "bootstrap_unit": "held-out episode",
    }


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
    target_mode: str = "full",
) -> dict[str, Any]:
    if target_mode not in {"full", "matched_band", "dynamics_target"}:
        raise ValueError(f"unsupported target mode: {target_mode}")
    dynamics_target_basis = bases["dynamics_basis"][:, :8]
    episode_ids = sorted({int(row["episode_id"]) for row in trajectories})
    rows: list[dict[str, Any]] = []
    # These are fixed before the folds are entered.  A one-step cyclic shift
    # breaks action/target alignment while preserving the action marginal.
    action_cache: dict[tuple[int, int], np.ndarray] = {}
    for row in trajectories:
        for horizon in HORIZONS:
            action_cache[(id(row), horizon)] = _action_features(
                actions, int(row["episode_id"]), int(row["start_observation"]), horizon,
                0, normalization_mode, action_mean, action_std,
            )
    all_action = [action_cache[(id(row), horizon)] for row in trajectories for horizon in HORIZONS]
    del all_action  # keep the cache keyed by row identity to avoid accidental reordering

    for test_episode in episode_ids:
        train_rows = [row for row in trajectories if int(row["episode_id"]) != test_episode]
        test_rows = [row for row in trajectories if int(row["episode_id"]) == test_episode]
        for basis_name, basis in bases.items():
            for band_name, (start, end) in BANDS.items():
                train_state = np.stack([
                    (row["true"][0] - mean) @ basis[:, start:end] for row in train_rows
                ])
                test_state = np.stack([
                    (row["true"][0] - mean) @ basis[:, start:end] for row in test_rows
                ])
                for horizon in HORIZONS:
                    if target_mode == "full":
                        train_target = np.stack([
                            row["true"][horizon] - row["true"][0] for row in train_rows
                        ])
                        test_target = np.stack([
                            row["true"][horizon] - row["true"][0] for row in test_rows
                        ])
                    elif target_mode == "matched_band":
                        train_target = np.stack([
                            (row["true"][horizon] - row["true"][0]) @ basis[:, start:end]
                            for row in train_rows
                        ])
                        test_target = np.stack([
                            (row["true"][horizon] - row["true"][0]) @ basis[:, start:end]
                            for row in test_rows
                        ])
                    else:
                        train_target = np.stack([
                            (row["true"][horizon] - row["true"][0]) @ dynamics_target_basis
                            for row in train_rows
                        ])
                        test_target = np.stack([
                            (row["true"][horizon] - row["true"][0]) @ dynamics_target_basis
                            for row in test_rows
                        ])
                    train_future = np.stack([action_cache[(id(row), horizon)] for row in train_rows])
                    test_future = np.stack([action_cache[(id(row), horizon)] for row in test_rows])
                    train_past = np.stack([
                        _action_features(actions, int(row["episode_id"]), int(row["start_observation"]), horizon,
                                         -horizon, normalization_mode, action_mean, action_std)
                        for row in train_rows
                    ])
                    test_past = np.stack([
                        _action_features(actions, int(row["episode_id"]), int(row["start_observation"]), horizon,
                                         -horizon, normalization_mode, action_mean, action_std)
                        for row in test_rows
                    ])
                    # The permutation is defined within the complete external
                    # window table, never from target values or held-out fits.
                    all_future = np.stack([
                        action_cache[(id(row), horizon)] for row in trajectories
                    ])
                    shift = np.roll(all_future, 1, axis=0)
                    index_by_id = {id(row): index for index, row in enumerate(trajectories)}
                    train_deranged = np.stack([shift[index_by_id[id(row)]] for row in train_rows])
                    test_deranged = np.stack([shift[index_by_id[id(row)]] for row in test_rows])
                    train_features = {
                        "action_future": train_future,
                        "action_past": train_past,
                        "action_deranged": train_deranged,
                        "state_only": train_state,
                        "state_plus_action": np.concatenate([train_state, train_future], axis=1),
                        "state_plus_past": np.concatenate([train_state, train_past], axis=1),
                        "state_plus_deranged": np.concatenate([train_state, train_deranged], axis=1),
                    }
                    test_features = {
                        "action_future": test_future,
                        "action_past": test_past,
                        "action_deranged": test_deranged,
                        "state_only": test_state,
                        "state_plus_action": np.concatenate([test_state, test_future], axis=1),
                        "state_plus_past": np.concatenate([test_state, test_past], axis=1),
                        "state_plus_deranged": np.concatenate([test_state, test_deranged], axis=1),
                    }
                    target_mean = train_target.mean(axis=0)
                    baseline = float(np.square(test_target - target_mean).mean())
                    for feature in FEATURES:
                        probe = _fit_probe(train_features[feature], train_target, ridge)
                        prediction = _decode(probe, test_features[feature])
                        mse = float(np.square(prediction - test_target).mean())
                        rows.append({
                            "episode_id": int(test_episode),
                            "basis": basis_name,
                            "band": band_name,
                            "horizon": int(horizon),
                            "feature": feature,
                            "mse": mse,
                            "r2": float(1.0 - mse / max(baseline, 1e-12)),
                            "baseline_mse": baseline,
                            "n_train_episodes": int(len(episode_ids) - 1),
                        })

    summary: dict[str, Any] = {}
    for basis_name in bases:
        for band_name in BANDS:
            for horizon in HORIZONS:
                for feature in FEATURES:
                    selected = [
                        row for row in rows
                        if row["basis"] == basis_name and row["band"] == band_name
                        and row["horizon"] == horizon and row["feature"] == feature
                    ]
                    values = np.asarray([row["r2"] for row in selected])
                    key = f"{basis_name}/{band_name}/h{horizon}/{feature}"
                    summary[key] = _bootstrap(values, bootstrap_repeats, 20261100 + horizon)
                for feature in ("state_plus_action", "state_only", "action_past", "action_deranged"):
                    selected = [
                        row for row in rows
                        if row["basis"] == basis_name and row["band"] == band_name
                        and row["horizon"] == horizon and row["feature"] == feature
                    ]
                    action = [
                        row for row in rows
                        if row["basis"] == basis_name and row["band"] == band_name
                        and row["horizon"] == horizon and row["feature"] == "action_future"
                    ]
                    contrast = np.asarray([a["r2"] - b["r2"] for a, b in zip(selected, action)])
                    key = f"{basis_name}/{band_name}/h{horizon}/{feature}_minus_action_future"
                    summary[key] = _bootstrap(contrast, bootstrap_repeats, 20261200 + horizon)
    return {"records": rows, "summary": summary}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]
    lines = [
        "# External forward readout of frozen latent dynamics",
        "",
        f"The target is `{report['protocol']['target']}`. Features use only the current latent coordinates and action summaries; all probes are leave-one-episode-out.",
        "",
        "| basis | band | horizon | action R2 | state+action R2 | state+action minus action | past-action minus action | deranged-action minus action |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for basis in report["protocol"]["bases"]:
        for band in BANDS:
            for horizon in HORIZONS:
                def fmt(feature: str) -> str:
                    value = summary[f"{basis}/{band}/h{horizon}/{feature}"]
                    return f"{value['mean']:.4g} [{value['ci95'][0]:.4g}, {value['ci95'][1]:.4g}]"
                lines.append(
                    f"| {basis} | {band} | {horizon} | {fmt('action_future')} | {fmt('state_plus_action')} | "
                    f"{fmt('state_plus_action_minus_action_future')} | {fmt('action_past_minus_action_future')} | "
                    f"{fmt('action_deranged_minus_action_future')} |"
                )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- Positive state-plus-action gain is an out-of-episode predictive readout of encoded latent motion, not a causal action effect.",
        "- Past-action and deranged-action controls test temporal alignment; they do not prove physical controllability.",
        "- The external archive has no intervention or online task-success endpoint in this audit.",
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
    ridge: float = 1e-2,
    bootstrap_repeats: int = 2000,
    random_seed: int = 20261004,
    target_mode: str = "full",
) -> dict[str, Any]:
    from experiments.io_utils import load_policy

    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, action_source = _normalization_parameters(actions, normalization_mode)
    trajectories = _load_trajectories(
        model, val_root, device, normalization_mode, windows_per_episode, max(HORIZONS)
    )
    for index, row in enumerate(trajectories):
        row["row_index"] = index
    mean, bases = _load_bases(basis_path, random_seed)
    crossfit = _crossfit(
        trajectories, actions, mean, bases, normalization_mode, action_mean, action_std,
        ridge, bootstrap_repeats, target_mode,
    )
    report = {
        "artifact": "external_frozen_forward_latent_dynamics_readout",
        "status": "complete",
        "source": {
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "basis_artifact": str(basis_path),
            "basis_sha256": _sha256(basis_path),
            "episodes": int(len({row["episode_id"] for row in trajectories})),
            "windows": int(len(trajectories)),
            "latent_dim": int(mean.shape[0]),
        },
        "protocol": {
            "target": {
                "full": "full encoded latent increment z[t+h]-z[t]",
                "matched_band": "future increment projected into the same source-fitted basis band",
                "dynamics_target": "future increment projected into the source-fitted dynamics basis top-8 target",
            }[target_mode],
            "features": list(FEATURES),
            "horizons": list(HORIZONS),
            "bands": BANDS,
            "bases": list(bases),
            "basis_fit": "separate frozen internal export; no external refitting",
            "probe_fit": "ridge on true encoded trajectories, leave-one-episode-out",
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": action_source,
            },
            "windows_per_episode": int(windows_per_episode),
            "ridge": float(ridge),
            "bootstrap_repeats": int(bootstrap_repeats),
            "bootstrap_unit": "held-out episode",
            "random_seed": int(random_seed),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "The future target is never supplied as a feature; this is a forward predictive readout rather than a projection-share statistic.",
            "Positive out-of-episode R2 or state-plus-action gain is evidence of predictive information in encoded latent motion, not a causal mechanism.",
            "Past-action and deranged-action controls test temporal alignment but do not establish physical counterfactual prediction or controllability.",
            "The audit does not establish online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("forward_dynamics_readout_notes.txt").write_text(
        "Forward latent-dynamics readout completed with future-target exclusion and episode-held-out probes.\n"
        "Interpretation is predictive latent evidence only.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--windows-per-episode", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20261004)
    parser.add_argument("--target-mode", choices=["full", "matched_band", "dynamics_target"], default="full")
    args = parser.parse_args()
    run_report(
        args.val_root, args.archive, args.basis, args.output, args.policy, args.device,
        args.action_normalization, args.windows_per_episode, args.ridge,
        args.bootstrap_repeats, args.random_seed, args.target_mode,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
