"""Cross-fitted action-residual transport of frozen latent dynamics geometry.

This audit starts from external encoded latent displacements and removes the
part predicted by a compact, explicitly declared action-history probe. The
action probe is fitted leave-one-episode-out, so the residual target for an
episode is never produced by a model fitted on that episode. Source-fitted
covariance and dynamics bases are then evaluated on the residual displacement
without external refitting.

The result is intentionally a bounded latent diagnostic. A weak action-only
readout does not prove that the remaining signal is physical or causal; it
only makes the action-conditioned latent comparison explicit.
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

from experiments.empirical_spectral_audit.run_external_dynamics_horizon_transport import (
    BASIS_NAMES,
    HORIZON_PAIRS,
    _basis_from_archive,
    _bootstrap_mean,
    _finite,
    _projection_share,
)


CONTEXT_LENGTH = 3
FRAME_STRIDE = 5
NORMALIZATIONS = ("public", "val_zscore", "raw_div100")
PUBLIC_ACTION_MEAN = np.asarray([-0.0087, 0.0068], dtype=np.float64)
PUBLIC_ACTION_STD = np.asarray([0.2019, 0.2002], dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_targets_and_windows(vectors_path: Path, metadata_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, int]:
    with np.load(vectors_path, allow_pickle=False) as archive:
        targets = {
            horizon: np.asarray(archive[f"observed/h{horizon}/target"], dtype=np.float64)
            for horizon in (1, 2, 4, 8)
        }
    if len({array.shape for array in targets.values()}) != 1:
        raise ValueError("external target arrays must have identical shapes")
    records = json.loads(metadata_path.read_text())["records"]
    observed = [row for row in records if row["mode"] == "observed" and int(row["horizon"]) == 1]
    groups = np.asarray([int(row["episode_id"]) for row in observed], dtype=np.int64)
    starts = np.asarray([int(row["start_observation"]) for row in observed], dtype=np.int64)
    row_count, latent_dim = next(iter(targets.values())).shape
    if groups.size != row_count:
        raise ValueError("metadata windows do not match external target arrays")
    return targets, groups, starts, int(latent_dim)


def _normalization_parameters(actions: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.asarray(actions, dtype=np.float64) / 100.0
    if mode == "public":
        return PUBLIC_ACTION_MEAN, PUBLIC_ACTION_STD
    if mode == "val_zscore":
        return scaled.mean(axis=(0, 1)), np.maximum(scaled.std(axis=(0, 1)), 1e-6)
    if mode == "raw_div100":
        return np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.float64)
    raise ValueError(f"unsupported normalization mode: {mode}")


def _action_features(
    actions: np.ndarray,
    episode_id: int,
    start_observation: int,
    first_horizon: int,
    second_horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = (actions[int(episode_id)] / 100.0 - mean) / std
    first_observation = int(start_observation) + CONTEXT_LENGTH - 1 + int(first_horizon)
    blocks = []
    for observation in range(first_observation, first_observation + int(second_horizon - first_horizon)):
        block = normalized[observation * FRAME_STRIDE : (observation + 1) * FRAME_STRIDE]
        if block.shape[0] < FRAME_STRIDE:
            block = np.pad(block, ((0, FRAME_STRIDE - block.shape[0]), (0, 0)))
        blocks.append(block.mean(axis=0))
    sequence = np.asarray(blocks, dtype=np.float64)
    # This compact summary is the same mean/first/last action-history probe
    # used by the existing physical-motion audit; it avoids an underdetermined
    # flattened action readout with only 21 external episodes.
    return np.concatenate([sequence.mean(axis=0), sequence[0], sequence[-1]])


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    alpha = float(ridge) * max(float(np.trace(gram)) / max(x.shape[1], 1), 1e-12)
    coefficient = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), centered_x.T @ centered_y)
    return coefficient, x_mean, y_mean


def _predict(fit: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coefficient, x_mean, y_mean = fit
    return (x - x_mean) @ coefficient + y_mean


def _crossfit_action_residual(
    displacement: np.ndarray,
    groups: np.ndarray,
    starts: np.ndarray,
    actions: np.ndarray,
    first_horizon: int,
    second_horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, float]:
    features = np.stack(
        [
            _action_features(actions, episode, start, first_horizon, second_horizon, mean, std)
            for episode, start in zip(groups, starts)
        ]
    )
    predictions = np.empty_like(displacement)
    for test_episode in np.unique(groups):
        test_mask = groups == test_episode
        train_mask = ~test_mask
        fit = _fit_ridge(features[train_mask], displacement[train_mask], ridge=ridge)
        predictions[test_mask] = _predict(fit, features[test_mask])
    residual = displacement - predictions
    denominator = max(float(np.square(displacement - displacement.mean(axis=0)).sum()), 1e-12)
    action_r2 = float(1.0 - np.square(residual).sum() / denominator)
    return residual, action_r2


def run_report(
    vectors_path: Path,
    metadata_path: Path,
    val_root: Path,
    basis_paths: list[Path],
    output: Path,
    ks: list[int],
    random_repeats: int,
    bootstrap_repeats: int,
    ridge: float,
    seed: int,
) -> dict[str, Any]:
    if len(basis_paths) < 2:
        raise ValueError("at least two source basis artifacts are required")
    actions = np.asarray(torch.load(val_root / "rel_actions.pth", map_location="cpu", weights_only=False), dtype=np.float64)
    with (val_root / "seq_lengths.pkl").open("rb") as handle:
        sequence_lengths = [int(value) for value in pickle.load(handle)]
    del sequence_lengths  # The source windows already encode valid start positions.
    targets, groups, starts, dim = _load_targets_and_windows(vectors_path, metadata_path)
    ks = sorted({int(k) for k in ks})
    if any(k <= 0 or k > dim for k in ks):
        raise ValueError(f"ks must lie in [1, {dim}]")

    source_bases: list[dict[str, np.ndarray]] = []
    for basis_path in basis_paths:
        with np.load(basis_path, allow_pickle=False) as archive:
            names = set(archive.files)
        source_bases.append(
            {
                name: _basis_from_archive(basis_path, name, dim)
                for name in BASIS_NAMES
                if name in names
            }
        )
    if any("covariance_basis" not in item or "dynamics_basis" not in item for item in source_bases):
        raise KeyError("each source basis must provide covariance_basis and dynamics_basis")

    action_probe: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for normalization in NORMALIZATIONS:
        action_mean, action_std = _normalization_parameters(actions, normalization)
        for first_horizon, second_horizon in HORIZON_PAIRS:
            pair = f"h{first_horizon}_to_h{second_horizon}"
            displacement = targets[second_horizon] - targets[first_horizon]
            residual, action_r2 = _crossfit_action_residual(
                displacement,
                groups,
                starts,
                actions,
                first_horizon,
                second_horizon,
                action_mean,
                action_std,
                ridge,
            )
            action_probe[f"{normalization}/{pair}"] = {
                "crossfit_action_r2": float(action_r2),
                "residual_energy_over_raw": float(np.square(residual).sum() / max(np.square(displacement).sum(), 1e-12)),
                "rows": int(residual.shape[0]),
                "episodes": int(np.unique(groups).size),
                "feature_definition": "mean, first, and last five-frame action-block means",
            }
            for source_index, bases in enumerate(source_bases):
                for k in ks:
                    random_values = []
                    for repeat in range(int(random_repeats)):
                        random = np.linalg.qr(
                            np.random.default_rng(seed + source_index * 1_000_000 + k * 10_000 + repeat).normal(size=(dim, k)),
                            mode="reduced",
                        )[0]
                        random_values.append(_projection_share(residual, random, k))
                    random_mean = np.stack(random_values, axis=0).mean(axis=0)
                    for basis_name, basis in bases.items():
                        values = _projection_share(residual, basis, k)
                        records.append(
                            {
                                "normalization": normalization,
                                "source_basis": str(basis_paths[source_index]),
                                "source_index": int(source_index),
                                "pair": pair,
                                "first_horizon": int(first_horizon),
                                "second_horizon": int(second_horizon),
                                "k": int(k),
                                "basis": basis_name,
                                "projection_share": _bootstrap_mean(
                                    values, groups, bootstrap_repeats, seed + 101 + source_index * 17 + k
                                ),
                                "delta_vs_covariance": _bootstrap_mean(
                                    values - _projection_share(residual, bases["covariance_basis"], k),
                                    groups,
                                    bootstrap_repeats,
                                    seed + 401 + source_index * 17 + k,
                                ),
                                "delta_vs_random": _bootstrap_mean(
                                    values - random_mean,
                                    groups,
                                    bootstrap_repeats,
                                    seed + 701 + source_index * 17 + k,
                                ),
                            }
                        )

    summary: dict[str, Any] = {}
    for normalization in NORMALIZATIONS:
        for first_horizon, second_horizon in HORIZON_PAIRS:
            pair = f"h{first_horizon}_to_h{second_horizon}"
            for k in ks:
                for basis in sorted({row["basis"] for row in records}):
                    rows = [
                        row for row in records
                        if row["normalization"] == normalization and row["pair"] == pair and row["k"] == k and row["basis"] == basis
                    ]
                    if not rows:
                        continue
                    covariance_deltas = [row["delta_vs_covariance"]["mean"] for row in rows]
                    random_deltas = [row["delta_vs_random"]["mean"] for row in rows]
                    summary[f"{normalization}/{basis}/{pair}/k{k}"] = {
                        "source_count": len(rows),
                        "projection_share_mean_across_sources": float(np.mean([row["projection_share"]["mean"] for row in rows])),
                        "delta_vs_covariance_mean_across_sources": float(np.mean(covariance_deltas)),
                        "delta_vs_covariance_range": [float(np.min(covariance_deltas)), float(np.max(covariance_deltas))],
                        "delta_vs_covariance_positive_all_sources": bool(all(value > 0.0 for value in covariance_deltas)),
                        "delta_vs_random_mean_across_sources": float(np.mean(random_deltas)),
                        "delta_vs_random_range": [float(np.min(random_deltas)), float(np.max(random_deltas))],
                        "delta_vs_random_positive_all_sources": bool(all(value > 0.0 for value in random_deltas)),
                        "source_rows": rows,
                    }

    claim_k = 8 if 8 in ks else ks[0]
    focus_keys = [f"{normalization}/dynamics_basis/h{a}_to_h{b}/k{claim_k}" for normalization in NORMALIZATIONS for a, b in HORIZON_PAIRS]
    report = {
        "artifact": "external_action_residual_horizon_transport",
        "status": "complete_crossfit_action_residual_transport",
        "source": {
            "vectors": str(vectors_path),
            "vectors_sha256": _sha256(vectors_path),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "validation_root": str(val_root),
            "rows": int(groups.size),
            "episodes": int(np.unique(groups).size),
            "latent_dim": int(dim),
            "basis_artifacts": [str(path) for path in basis_paths],
            "basis_sha256": [_sha256(path) for path in basis_paths],
        },
        "protocol": {
            "horizon_pairs": [list(pair) for pair in HORIZON_PAIRS],
            "normalizations": list(NORMALIZATIONS),
            "k_values": ks,
            "basis_fit": "source-only frozen latent exports; no external refitting",
            "action_probe": "leave-one-episode-out ridge on compact action-history summary",
            "action_probe_ridge": float(ridge),
            "random_subspace_repeats": int(random_repeats),
            "bootstrap_repeats": int(bootstrap_repeats),
            "bootstrap_unit": "episode",
            "interpretation": "encoded latent displacement after cross-fitted action readout",
        },
        "action_probe": action_probe,
        "records": records,
        "summary": summary,
        "claim_matrix": [
            {
                "claim": "Dynamics-basis enrichment survives cross-fitted action residualization in the encoded latent target.",
                "evidence": "dynamics basis exceeds covariance and random controls for every horizon pair, source split, and action-normalization arm at top-k",
                "verdict": "supported as action-conditioned latent diagnostic" if all(
                    summary[key]["delta_vs_covariance_positive_all_sources"] and summary[key]["delta_vs_random_positive_all_sources"]
                    for key in focus_keys
                ) else "partial or not supported",
            },
            {
                "claim": "The residual projection establishes extra physical-motion information beyond actions.",
                "evidence": "the residual target is an encoded latent displacement and the compact action probe is only a predictive control",
                "verdict": "not supported by this audit",
            },
            {
                "claim": "The result is a causal controllability or planning improvement.",
                "evidence": "no environment intervention, retraining, online planner, or task-success endpoint is evaluated",
                "verdict": "not supported by this audit",
            },
        ],
        "claim_boundary": [
            "Action residuals are cross-fitted by episode; no action readout is fitted on the held-out episode.",
            "The compact action probe is a declared baseline, not a sufficient statistic for all action effects.",
            "Projection enrichment concerns encoded latent displacement and does not establish physical observability, causality, online planning, task success, or retraining gain.",
            "Normalization arms are reported separately; positivity here does not imply invariance of the underlying checkpoint or training statistics.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cross-fitted action-residual latent dynamics transport",
        "",
        "External encoded latent displacements are residualized by a compact action-history ridge probe fitted leave-one-episode-out. Source-fitted bases are then evaluated without external refitting.",
        "",
        "## Action probe",
        "",
        "| normalization | displacement | cross-fit action R² | residual/raw energy |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, value in report["action_probe"].items():
        normalization, pair = key.split("/", 1)
        lines.append(f"| {normalization} | {pair} | {value['crossfit_action_r2']:.6f} | {value['residual_energy_over_raw']:.6f} |")
    lines.extend([
        "",
        f"## Dynamics basis on action residuals (top-{8 if 8 in report['protocol']['k_values'] else report['protocol']['k_values'][0]})",
        "",
        "| normalization | displacement | share | vs covariance | vs random | all source signs positive |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ])
    display_k = 8 if 8 in report["protocol"]["k_values"] else report["protocol"]["k_values"][0]
    for normalization in NORMALIZATIONS:
        for first_horizon, second_horizon in HORIZON_PAIRS:
            key = f"{normalization}/dynamics_basis/h{first_horizon}_to_h{second_horizon}/k{display_k}"
            value = report["summary"][key]
            lines.append(
                f"| {normalization} | h{first_horizon}_to_h{second_horizon} | {value['projection_share_mean_across_sources']:.6f} "
                f"| {value['delta_vs_covariance_mean_across_sources']:+.6f} | {value['delta_vs_random_mean_across_sources']:+.6f} "
                f"| {value['delta_vs_covariance_positive_all_sources'] and value['delta_vs_random_positive_all_sources']} |"
            )
    lines.extend(["", "## Claim matrix", "", "| claim | evidence | verdict |", "| --- | --- | --- |"])
    for row in report["claim_matrix"]:
        lines.append(f"| {row['claim']} | {row['evidence']} | {row['verdict']} |")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--basis", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--random-repeats", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    run_report(
        vectors_path=args.vectors,
        metadata_path=args.metadata,
        val_root=args.val_root,
        basis_paths=args.basis,
        output=args.output,
        ks=args.ks,
        random_repeats=args.random_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        ridge=args.ridge,
        seed=args.seed,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
