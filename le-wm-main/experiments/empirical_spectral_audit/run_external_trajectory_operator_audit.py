"""Trajectory-level operator audit for the frozen LeWM PushT checkpoint.

The existing external audits establish multi-horizon prediction error and
spectral-band intervention effects.  This audit adds a trajectory-level
operator diagnostic: it reconstructs the full eight-step true and
autoregressive latent trajectories for each external window, fits transition
operators on other episodes, and measures spectral radius, singular-value
growth, non-normality, and direct-vs-composed operator defects.

All operators are fitted in a low-dimensional PCA coordinate system learned on
the training episodes of each leave-one-episode-out fold.  This prevents the
192-dimensional, 42-window external sample from being treated as a
well-identified full-rank operator.  The artifact is frozen external evidence
only; it is not an environment intervention, online control, planning result,
or retraining experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_lewm_external_pusht_temporal_audit import (  # noqa: E402
    CONTEXT_LENGTH,
    FRAME_STRIDE,
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    _load_video_frames,
    _make_action_blocks,
    _window_starts,
)


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


def _pca_fit(states: np.ndarray, dimension: int) -> dict[str, np.ndarray]:
    if states.ndim != 3 or states.shape[0] < 2 or states.shape[1] < 2:
        raise ValueError("states must have shape (trajectories, time, latent_dim)")
    flattened = states.reshape(-1, states.shape[-1]).astype(np.float64, copy=False)
    mean = flattened.mean(axis=0)
    centered = flattened - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular_values > max(float(singular_values[0]) * 1e-10, 1e-12)))
    usable = max(1, min(int(dimension), rank, centered.shape[0] - 1, centered.shape[1]))
    return {
        "mean": mean,
        "basis": right_vectors[:usable].T.copy(),
        "rank": np.asarray(rank, dtype=np.int64),
    }


def _pca_transform(states: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(states, dtype=np.float64) - fit["mean"]) @ fit["basis"]


def _fit_row_operator(source: np.ndarray, target: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must have matching two-dimensional shapes")
    if source.shape[0] < 2:
        raise ValueError("at least two rows are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    denominator = float(max(source.shape[0] - 1, 1))
    covariance = x.T @ x / denominator
    cross = x.T @ y / denominator
    scale = max(float(np.trace(covariance)) / source.shape[1], 1e-8)
    row_operator = np.linalg.solve(
        covariance + float(ridge) * scale * np.eye(source.shape[1]), cross
    )
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "row_operator": row_operator,
    }


def _apply_row_operator(fit: dict[str, np.ndarray], source: np.ndarray) -> np.ndarray:
    return (source - fit["source_mean"]) @ fit["row_operator"] + fit["target_mean"]


def _operator_diagnostics(row_operator: np.ndarray, horizons: tuple[int, ...]) -> dict[str, Any]:
    operator = np.asarray(row_operator, dtype=np.float64).T
    norm = max(float(np.linalg.norm(operator, ord="fro")), 1e-12)
    nonnormality = np.linalg.norm(operator.T @ operator - operator @ operator.T, ord="fro") / (norm**2)
    eigenvalues = np.linalg.eigvals(operator)
    radius = max(float(np.max(np.abs(eigenvalues))), 1e-12)
    singular_values = np.linalg.svd(operator, compute_uv=False)
    powers: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        power = np.linalg.matrix_power(operator, int(horizon))
        gain = float(np.linalg.svd(power, compute_uv=False)[0])
        powers[str(horizon)] = {
            "max_singular_value": gain,
            "gain_over_radius_power": gain / max(radius**int(horizon), 1e-12),
        }
    return {
        "dimension": int(operator.shape[0]),
        "spectral_radius": radius,
        "max_singular_value": float(singular_values[0]),
        "singular_to_radius_ratio": float(singular_values[0] / radius),
        "henrici_nonnormality": float(nonnormality),
        "operator_frobenius_norm": float(norm),
        "operator_powers": powers,
    }


def _operator_composition_defect(
    transition: dict[str, np.ndarray],
    direct: dict[str, np.ndarray],
    horizon: int,
) -> float:
    composed = np.linalg.matrix_power(transition["row_operator"].T, int(horizon))
    direct_operator = direct["row_operator"].T
    denominator = max(float(np.linalg.norm(direct_operator, ord="fro")), 1e-12)
    return float(np.linalg.norm(composed - direct_operator, ord="fro") / denominator)


def _fit_transition_family(states: np.ndarray, horizons: tuple[int, ...], ridge: float) -> dict[str, Any]:
    trajectories, time_steps, dimension = states.shape
    if time_steps < max(horizons) + 1:
        raise ValueError("states do not contain all requested horizons")
    transition = _fit_row_operator(
        states[:, :-1].reshape(-1, dimension),
        states[:, 1:].reshape(-1, dimension),
        ridge=ridge,
    )
    direct: dict[str, dict[str, np.ndarray]] = {}
    for horizon in horizons:
        direct[str(horizon)] = _fit_row_operator(
            states[:, :-horizon].reshape(-1, dimension),
            states[:, horizon:].reshape(-1, dimension),
            ridge=ridge,
        )
    return {"transition": transition, "direct": direct}


def _evaluate_family(
    family: dict[str, Any],
    test_states: np.ndarray,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    dimension = test_states.shape[-1]
    transition = family["transition"]
    predicted_next = _apply_row_operator(
        transition, test_states[:, :-1].reshape(-1, dimension)
    )
    true_next = test_states[:, 1:].reshape(-1, dimension)
    mse = float(np.square(predicted_next - true_next).mean())
    target_mse = float(np.square(true_next - transition["target_mean"]).mean())
    result: dict[str, Any] = {
        "one_step_mse": mse,
        "one_step_r2": float(1.0 - mse / max(target_mse, 1e-12)),
        "operator": _operator_diagnostics(transition["row_operator"], horizons),
        "direct_operator": {},
        "composition_defect": {},
    }
    for horizon in horizons:
        direct = family["direct"][str(horizon)]
        source = test_states[:, :-horizon].reshape(-1, dimension)
        target = test_states[:, horizon:].reshape(-1, dimension)
        direct_prediction = _apply_row_operator(direct, source)
        direct_mse = float(np.square(direct_prediction - target).mean())
        direct_target_mse = float(np.square(target - direct["target_mean"]).mean())
        result["direct_operator"][str(horizon)] = {
            "mse": direct_mse,
            "r2": float(1.0 - direct_mse / max(direct_target_mse, 1e-12)),
            "diagnostics": _operator_diagnostics(direct["row_operator"], horizons),
        }
        result["composition_defect"][str(horizon)] = _operator_composition_defect(
            transition, direct, horizon
        )
    return result


def _bootstrap_clustered(
    values: np.ndarray, groups: np.ndarray, repeats: int, seed: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.asarray(sorted(set(groups.tolist())))
    if values.ndim != 1 or groups.shape != values.shape or unique.size < 2:
        raise ValueError("cluster bootstrap requires paired values and at least two groups")
    cluster_means = np.asarray([values[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    draws = cluster_means[rng.integers(0, cluster_means.size, size=(int(repeats), cluster_means.size))]
    means = draws.mean(axis=1)
    return {
        "n_windows": int(values.size),
        "clusters": int(unique.size),
        "mean": float(cluster_means.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_repeats": int(repeats),
        "bootstrap_unit": "episode",
    }


def _predict_full_trajectory(
    model: torch.nn.Module,
    embedding: torch.Tensor,
    action_embedding: torch.Tensor,
    max_horizon: int,
) -> torch.Tensor:
    context_emb = embedding[:, :CONTEXT_LENGTH].clone()
    context_act = action_embedding[:, :CONTEXT_LENGTH].clone()
    states = [embedding[:, CONTEXT_LENGTH - 1].detach().cpu()]
    with torch.inference_mode():
        for step in range(1, max_horizon + 1):
            prediction = model.predict(context_emb, context_act)[:, -1]
            states.append(prediction.detach().cpu())
            if step < max_horizon:
                next_action = action_embedding[
                    :, CONTEXT_LENGTH - 1 + step : CONTEXT_LENGTH + step
                ]
                context_emb = torch.cat((context_emb, prediction[:, None]), dim=1)[
                    :, -CONTEXT_LENGTH:
                ]
                context_act = torch.cat((context_act, next_action), dim=1)[
                    :, -CONTEXT_LENGTH:
                ]
    return torch.stack(states, dim=1)


def _normalization_parameters(
    actions: torch.Tensor, normalization_mode: str
) -> tuple[torch.Tensor, torch.Tensor, str]:
    scaled = actions.float() / 100.0
    if normalization_mode == "public":
        return (
            PUBLIC_ACTION_MEAN,
            PUBLIC_ACTION_STD,
            "public JEPA-WM PushT loader constants",
        )
    if normalization_mode == "val_zscore":
        return (
            scaled.mean(dim=(0, 1)),
            scaled.std(dim=(0, 1)).clamp_min(1e-6),
            "statistics computed from the external validation actions",
        )
    if normalization_mode == "raw_div100":
        return torch.zeros(2), torch.ones(2), "unit scale after raw action division by 100"
    raise ValueError(f"unsupported normalization mode: {normalization_mode}")


def _load_trajectories(
    model: torch.nn.Module,
    val_root: Path,
    device: torch.device,
    normalization_mode: str,
    windows_per_episode: int,
    max_horizon: int,
) -> list[dict[str, Any]]:
    actions = torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False).float()
    action_mean, action_std, _ = _normalization_parameters(actions, normalization_mode)
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        lengths = [int(value) for value in pickle.load(handle)]
    from utils import get_img_preprocessor

    preprocess = get_img_preprocessor("pixels", "pixels", 224)
    trajectories: list[dict[str, Any]] = []
    for episode_id, length in enumerate(lengths):
        starts = _window_starts(
            length,
            context_length=CONTEXT_LENGTH,
            max_horizon=max_horizon,
            windows_per_episode=windows_per_episode,
        )
        print(f"episode={episode_id} length={length} windows={len(starts)}", flush=True)
        for start in starts:
            count = CONTEXT_LENGTH + max_horizon
            frame_indices = [FRAME_STRIDE * (start + i) for i in range(count)]
            frames = _load_video_frames(
                val_root / "obses" / f"episode_{episode_id:03d}.mp4", frame_indices
            )
            pixels = preprocess({"pixels": frames})["pixels"].unsqueeze(0).to(device)
            action_blocks = _make_action_blocks(
                actions[episode_id],
                start,
                count,
                "observed",
                normalization_mode=normalization_mode,
                action_mean=action_mean,
                action_std=action_std,
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                encoded = model.encode({"pixels": pixels, "action": action_blocks})
                predicted = _predict_full_trajectory(
                    model, encoded["emb"], encoded["act_emb"], max_horizon
                )
            true = encoded["emb"][:, CONTEXT_LENGTH - 1 : CONTEXT_LENGTH + max_horizon]
            trajectories.append(
                {
                    "episode_id": int(episode_id),
                    "start_observation": int(start),
                    "true": true.detach().cpu().numpy()[0].astype(np.float64),
                    "predicted": predicted.numpy()[0].astype(np.float64),
                    # One two-dimensional summary per latent transition.  The
                    # block at CONTEXT_LENGTH-1+t is the action supplied to
                    # the predictor for z_t -> z_{t+1}; retaining it makes
                    # the trajectory artifact reusable for action-algebra
                    # audits without rerunning the encoder.
                    "action_features": action_blocks[0,
                        CONTEXT_LENGTH - 1 : CONTEXT_LENGTH - 1 + max_horizon
                    ].reshape(max_horizon, FRAME_STRIDE, -1).mean(dim=1).detach().cpu().numpy().astype(np.float64),
                }
            )
    return trajectories


def _crossfit(
    trajectories: list[dict[str, Any]],
    dimension: int,
    ridge: float,
    horizons: tuple[int, ...],
    bootstrap_repeats: int,
) -> dict[str, Any]:
    episode_ids = np.asarray(sorted({row["episode_id"] for row in trajectories}), dtype=np.int64)
    fold_records: list[dict[str, Any]] = []
    for test_episode in episode_ids.tolist():
        train_rows = [row for row in trajectories if row["episode_id"] != test_episode]
        test_rows = [row for row in trajectories if row["episode_id"] == test_episode]
        fold: dict[str, Any] = {"test_episode": int(test_episode), "windows": len(test_rows)}
        for source_name in ("true", "predicted"):
            train_states = np.stack([row[source_name] for row in train_rows])
            test_states = np.stack([row[source_name] for row in test_rows])
            pca = _pca_fit(
                np.concatenate((
                    np.stack([row["true"] for row in train_rows]),
                    np.stack([row["predicted"] for row in train_rows]),
                )),
                dimension,
            )
            train_reduced = _pca_transform(train_states, pca)
            test_reduced = _pca_transform(test_states, pca)
            family = _fit_transition_family(train_reduced, horizons, ridge)
            evaluated = _evaluate_family(family, test_reduced, horizons)
            fold[source_name] = {
                "pca_dimension": int(pca["basis"].shape[1]),
                "pca_rank": int(pca["rank"]),
                "evaluation": evaluated,
            }
        fold_records.append(fold)

    summary: dict[str, Any] = {"operator_by_source": {}, "paired_operator_difference": {}, "error_amplification": {}}
    for source_name in ("true", "predicted"):
        summary["operator_by_source"][source_name] = {}
        for metric in ("spectral_radius", "max_singular_value", "singular_to_radius_ratio", "henrici_nonnormality"):
            summary["operator_by_source"][source_name][metric] = {
                "mean": float(np.mean([
                    row[source_name]["evaluation"]["operator"][metric]
                    for row in fold_records
                ])),
                "std": float(np.std([
                    row[source_name]["evaluation"]["operator"][metric]
                    for row in fold_records
                ], ddof=1)),
            }
        summary["operator_by_source"][source_name]["composition_defect"] = {
            str(h): {
                "mean": float(np.mean([
                    row[source_name]["evaluation"]["composition_defect"][str(h)]
                    for row in fold_records
                ])),
                "std": float(np.std([
                    row[source_name]["evaluation"]["composition_defect"][str(h)]
                    for row in fold_records
                ], ddof=1)),
            }
            for h in horizons
        }
        summary["operator_by_source"][source_name]["one_step_r2"] = {
            "mean": float(np.mean([
                row[source_name]["evaluation"]["one_step_r2"] for row in fold_records
            ])),
            "std": float(np.std([
                row[source_name]["evaluation"]["one_step_r2"] for row in fold_records
            ], ddof=1)),
        }
        summary["operator_by_source"][source_name]["power_gain"] = {
            str(h): {
                "mean": float(np.mean([
                    row[source_name]["evaluation"]["operator"]["operator_powers"][str(h)]["max_singular_value"]
                    for row in fold_records
                ])),
                "std": float(np.std([
                    row[source_name]["evaluation"]["operator"]["operator_powers"][str(h)]["max_singular_value"]
                    for row in fold_records
                ], ddof=1)),
            }
            for h in horizons
        }
        summary["operator_by_source"][source_name]["power_gain_over_radius"] = {
            str(h): {
                "mean": float(np.mean([
                    row[source_name]["evaluation"]["operator"]["operator_powers"][str(h)]["gain_over_radius_power"]
                    for row in fold_records
                ])),
                "std": float(np.std([
                    row[source_name]["evaluation"]["operator"]["operator_powers"][str(h)]["gain_over_radius_power"]
                    for row in fold_records
                ], ddof=1)),
            }
            for h in horizons
        }
    for h in horizons:
        true_values = np.asarray([
            row["true"]["evaluation"]["composition_defect"][str(h)] for row in fold_records
        ])
        pred_values = np.asarray([
            row["predicted"]["evaluation"]["composition_defect"][str(h)] for row in fold_records
        ])
        summary["paired_operator_difference"][f"composition_defect/h{h}"] = _bootstrap_clustered(
            pred_values - true_values,
            np.asarray([row["test_episode"] for row in fold_records]),
            bootstrap_repeats,
            20261003 + h,
        )
    for h in range(1, max(horizons) + 1):
        errors = np.asarray([
            np.square(row["true"][h] - row["predicted"][h]).mean()
            for row in trajectories
        ])
        base = np.asarray([
            np.square(row["true"][1] - row["predicted"][1]).mean()
            for row in trajectories
        ])
        ratios = errors / np.maximum(base, 1e-12)
        groups = np.asarray([row["episode_id"] for row in trajectories])
        summary["error_amplification"][f"h{h}"] = {
            "mse": _bootstrap_clustered(errors, groups, bootstrap_repeats, 20262003 + h),
            "ratio_to_h1": _bootstrap_clustered(ratios, groups, bootstrap_repeats, 20263003 + h),
            "fraction_above_h1": float(np.mean(errors > base)),
        }
    return {"folds": fold_records, "summary": summary}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["crossfit"]["summary"]
    lines = [
        "# External frozen LeWM trajectory-level operator audit",
        "",
        "Operators are fitted in a fold-specific PCA coordinate system using all training episodes and evaluated on the held-out episode. Values below are mean ± standard deviation across the 21 leave-one-episode-out folds unless stated otherwise.",
        "",
        "| source trajectory | one-step held-out R² | spectral radius | max singular value | sigma/radius | Henrici non-normality |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ("true", "predicted"):
        op = summary["operator_by_source"][source]
        fmt = lambda key: f"{op[key]['mean']:.6g} ± {op[key]['std']:.3g}"
        lines.append(
            f"| {source} | {fmt('one_step_r2')} | {fmt('spectral_radius')} | {fmt('max_singular_value')} | "
            f"{fmt('singular_to_radius_ratio')} | {fmt('henrici_nonnormality')} |"
        )
    lines.extend([
        "",
        "| source trajectory | h=2 gain / radius² | h=4 gain / radius⁴ | h=8 gain / radius⁸ |",
        "| --- | ---: | ---: | ---: |",
    ])
    for source in ("true", "predicted"):
        gains = summary["operator_by_source"][source]["power_gain_over_radius"]
        lines.append(
            f"| {source} | {gains['2']['mean']:.6g} ± {gains['2']['std']:.3g} | "
            f"{gains['4']['mean']:.6g} ± {gains['4']['std']:.3g} | "
            f"{gains['8']['mean']:.6g} ± {gains['8']['std']:.3g} |"
        )
    lines.extend([
        "",
        "| horizon | true composition defect | predicted composition defect | predicted − true (95% episode-bootstrap CI) | error MSE | error / h=1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for horizon in report["protocol"]["horizons"]:
        true_defect = summary["operator_by_source"]["true"]["composition_defect"][str(horizon)]
        pred_defect = summary["operator_by_source"]["predicted"]["composition_defect"][str(horizon)]
        diff = summary["paired_operator_difference"][f"composition_defect/h{horizon}"]
        err = summary["error_amplification"][f"h{horizon}"]
        lines.append(
            f"| {horizon} | {true_defect['mean']:.6g} ± {true_defect['std']:.3g} | "
            f"{pred_defect['mean']:.6g} ± {pred_defect['std']:.3g} | {diff['mean']:.6g} "
            f"[{diff['ci95'][0]:.6g}, {diff['ci95'][1]:.6g}] | {err['mse']['mean']:.6g} "
            f"[{err['mse']['ci95'][0]:.6g}, {err['mse']['ci95'][1]:.6g}] | {err['ratio_to_h1']['mean']:.6g} "
            f"[{err['ratio_to_h1']['ci95'][0]:.6g}, {err['ratio_to_h1']['ci95'][1]:.6g}] |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The operator is a low-dimensional, fold-fitted summary of frozen latent trajectories. It does not identify the nonlinear LeWM Jacobian, establish causal controllability, online planning, task success, retraining improvement, or a universal world-model property.",
        "",
        "The composition defect compares a fitted one-step operator power with a separately fitted direct h-step operator; it is a consistency diagnostic, not a causal decomposition of rollout error.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run_report(
    val_root: Path,
    archive: Path,
    policy: str,
    output: Path,
    horizons: tuple[int, ...] = (1, 2, 4, 8),
    windows_per_episode: int = 2,
    dimension: int = 16,
    ridge: float = 1e-2,
    device_name: str = "cpu",
    normalization_mode: str = "public",
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    if not (val_root / "states.pth").exists() or not (val_root / "rel_actions.pth").exists():
        raise FileNotFoundError("public validation root lacks states.pth or rel_actions.pth")
    if tuple(sorted(set(horizons))) != tuple(horizons) or min(horizons) < 1:
        raise ValueError("horizons must be sorted positive integers")
    if max(horizons) != 8:
        raise ValueError("the external protocol requires max_horizon=8")
    from experiments.io_utils import load_policy

    device = torch.device(device_name)
    model = load_policy(policy, device=str(device))
    action_values = torch.load(
        val_root / "rel_actions.pth", map_location="cpu", weights_only=False
    ).float()
    action_mean, action_std, action_source = _normalization_parameters(
        action_values, normalization_mode
    )
    trajectories = _load_trajectories(
        model,
        val_root,
        device,
        normalization_mode,
        windows_per_episode,
        max(horizons),
    )
    crossfit = _crossfit(
        trajectories,
        dimension=dimension,
        ridge=ridge,
        horizons=horizons,
        bootstrap_repeats=bootstrap_repeats,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact": "external_frozen_lewm_trajectory_operator_audit",
        "status": "frozen_external_trajectory_operator_diagnostic",
        "source": {
            "archive": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "archive_sha256": _sha256(archive),
            "validation_root": str(val_root),
            "episodes": int(len({row["episode_id"] for row in trajectories})),
            "windows": int(len(trajectories)),
        },
        "model": {"policy": policy, "device": str(device)},
        "protocol": {
            "context_length": CONTEXT_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "horizons": [int(h) for h in horizons],
            "windows_per_episode": int(windows_per_episode),
            "trajectory": "true encoded states versus autoregressive predicted states, both starting at the last context embedding",
            "operator_fit": "ridge row operator on all consecutive training-episode pairs",
            "crossfit": "leave-one-episode-out",
            "pca_dimension_requested": int(dimension),
            "ridge": float(ridge),
            "action_normalization": {
                "mode": normalization_mode,
                "raw_action_divisor": 100.0,
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "source": action_source,
                "original_lewm_training_statistics_available": False,
            },
            "bootstrap": "episode-cluster bootstrap for paired window-level error and composition contrasts",
            "bootstrap_repeats": int(bootstrap_repeats),
        },
        "crossfit": crossfit,
        "claim_boundary": [
            "This is a frozen external temporal diagnostic of the released pusht/lewm checkpoint.",
            "PCA reduction and ridge regularization are required by the 192-dimensional, 42-window protocol; full-rank nonlinear Jacobian identification is not claimed.",
            "Operator spectra, composition defects, and error amplification are descriptive diagnostics and do not establish causal controllability, online planning, task success, or retraining improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(
        "Trajectory-level frozen external operator audit completed.\n"
        "The report intentionally separates fitted operator geometry from causal control and planning claims.\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--pca-dimension", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--action-normalization", choices=["public", "val_zscore", "raw_div100"], default="public")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    started = time.time()
    report = run_report(
        val_root=args.val_root,
        archive=args.archive,
        policy=args.policy,
        output=args.output,
        horizons=tuple(args.horizons),
        windows_per_episode=args.windows_per_episode,
        dimension=args.pca_dimension,
        ridge=args.ridge,
        device_name=args.device,
        normalization_mode=args.action_normalization,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    report["protocol_elapsed_seconds"] = float(time.time() - started)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
