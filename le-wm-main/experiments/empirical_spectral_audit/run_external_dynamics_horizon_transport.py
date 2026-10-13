"""Test whether a frozen dynamics basis captures external multi-horizon latent motion.

The basis is fitted on an internal frozen latent export.  The external public
PushT vectors contain encoded future embeddings at several horizons; adjacent
horizon differences therefore provide held-out latent displacements without
refitting on the external trajectories.  We compare projection-energy shares
for the dynamics basis against covariance, residual-sensitive, conditional
dynamics, and dimension-matched random subspaces.

This is a latent-trajectory transport diagnostic.  It is not a physical-state
decoder, causal intervention, online-control evaluation, or task-success
measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HORIZON_PAIRS = ((1, 2), (2, 4), (4, 8))
BASIS_NAMES = (
    "covariance_basis",
    "dynamics_basis",
    "conditional_dynamics_basis",
    "residual_basis",
    "residual_covariance_basis",
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


def _bootstrap_mean(values: np.ndarray, groups: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if values.ndim != 1 or groups.shape != values.shape or unique.size < 2:
        raise ValueError("cluster bootstrap requires one value per row and at least two groups")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    by_group = [values[groups == group] for group in unique]
    draws = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        selected = rng.integers(0, len(by_group), size=len(by_group))
        draws[index] = float(np.concatenate([by_group[item] for item in selected]).mean())
    return {
        "mean": observed,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "clusters": int(unique.size),
        "rows": int(values.size),
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "episode",
    }


def _basis_from_archive(path: Path, name: str, latent_dim: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if name not in archive.files:
            raise KeyError(f"{path}: missing {name}")
        basis = np.asarray(archive[name], dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != latent_dim:
        raise ValueError(f"{path}:{name} has incompatible shape {basis.shape}")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), atol=2e-5, rtol=2e-5):
        raise ValueError(f"{path}:{name} is not orthonormal")
    return basis


def _random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(dim, k))
    return np.linalg.qr(matrix, mode="reduced")[0]


def _projection_share(displacements: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    projected = displacements @ basis[:, :k]
    numerator = np.sum(projected * projected, axis=1)
    denominator = np.maximum(np.sum(displacements * displacements, axis=1), 1e-12)
    return numerator / denominator


def _load_external_vectors(vectors_path: Path, metadata_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    with np.load(vectors_path, allow_pickle=False) as archive:
        targets = {
            horizon: np.asarray(archive[f"observed/h{horizon}/target"], dtype=np.float64)
            for horizon in (1, 2, 4, 8)
        }
    if len({array.shape for array in targets.values()}) != 1:
        raise ValueError("external target arrays must have identical shapes")
    rows = json.loads(metadata_path.read_text())["records"]
    observed = [row for row in rows if row["mode"] == "observed" and int(row["horizon"]) == 1]
    groups = np.asarray([int(row["episode_id"]) for row in observed], dtype=np.int64)
    if groups.shape[0] != next(iter(targets.values())).shape[0]:
        raise ValueError("metadata episode rows do not match external target rows")
    return targets, groups, int(next(iter(targets.values())).shape[1])


def run_report(
    vectors_path: Path,
    metadata_path: Path,
    basis_paths: list[Path],
    output: Path,
    ks: list[int],
    random_repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    if len(basis_paths) < 2:
        raise ValueError("at least two independent source basis artifacts are required")
    targets, groups, dim = _load_external_vectors(vectors_path, metadata_path)
    ks = sorted({int(k) for k in ks})
    if any(k <= 0 or k > dim for k in ks):
        raise ValueError(f"ks must lie in [1, {dim}]")

    source_records: list[dict[str, Any]] = []
    for source_index, basis_path in enumerate(basis_paths):
        available = []
        with np.load(basis_path, allow_pickle=False) as archive:
            available = list(archive.files)
        bases = {
            name: _basis_from_archive(basis_path, name, dim)
            for name in BASIS_NAMES
            if name in available
        }
        if "covariance_basis" not in bases or "dynamics_basis" not in bases:
            raise KeyError(f"{basis_path}: both covariance_basis and dynamics_basis are required")
        for first_horizon, second_horizon in HORIZON_PAIRS:
            displacement = targets[second_horizon] - targets[first_horizon]
            pair_name = f"h{first_horizon}_to_h{second_horizon}"
            for k in ks:
                shares = {
                    name: _projection_share(displacement, basis, k)
                    for name, basis in bases.items()
                }
                random_rows = []
                for repeat in range(int(random_repeats)):
                    random = _random_basis(dim, k, seed + source_index * 1_000_000 + k * 10_000 + repeat)
                    random_rows.append(_projection_share(displacement, random, k))
                random_matrix = np.stack(random_rows, axis=0)
                random_mean = random_matrix.mean(axis=0)
                for basis_name, values in shares.items():
                    row = {
                        "source_basis": str(basis_path),
                        "source_index": int(source_index),
                        "pair": pair_name,
                        "first_horizon": int(first_horizon),
                        "second_horizon": int(second_horizon),
                        "span": int(second_horizon - first_horizon),
                        "k": int(k),
                        "basis": basis_name,
                        "projection_share": _bootstrap_mean(
                            values, groups, bootstrap_repeats, seed + 101 + source_index * 17 + k
                        ),
                        "delta_vs_covariance": _bootstrap_mean(
                            values - shares["covariance_basis"],
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
                    source_records.append(row)

    summary: dict[str, Any] = {}
    for pair_name in [f"h{a}_to_h{b}" for a, b in HORIZON_PAIRS]:
        for k in ks:
            for basis_name in sorted({row["basis"] for row in source_records}):
                rows = [
                    row for row in source_records
                    if row["pair"] == pair_name and row["k"] == k and row["basis"] == basis_name
                ]
                if not rows:
                    continue
                deltas_cov = [row["delta_vs_covariance"]["mean"] for row in rows]
                deltas_random = [row["delta_vs_random"]["mean"] for row in rows]
                key = f"{basis_name}/{pair_name}/k{k}"
                summary[key] = {
                    "source_count": len(rows),
                    "projection_share_mean_across_sources": float(np.mean([row["projection_share"]["mean"] for row in rows])),
                    "projection_share_range": [float(np.min([row["projection_share"]["mean"] for row in rows])), float(np.max([row["projection_share"]["mean"] for row in rows]))],
                    "delta_vs_covariance_mean_across_sources": float(np.mean(deltas_cov)),
                    "delta_vs_covariance_range": [float(np.min(deltas_cov)), float(np.max(deltas_cov))],
                    "delta_vs_covariance_positive_all_sources": bool(all(value > 0.0 for value in deltas_cov)),
                    "delta_vs_random_mean_across_sources": float(np.mean(deltas_random)),
                    "delta_vs_random_range": [float(np.min(deltas_random)), float(np.max(deltas_random))],
                    "delta_vs_random_positive_all_sources": bool(all(value > 0.0 for value in deltas_random)),
                    "source_rows": rows,
                }

    claim_k = 8 if 8 in ks else ks[0]
    report = {
        "artifact": "external_dynamics_horizon_transport",
        "status": "complete_external_latent_displacement_transport",
        "source": {
            "vectors": str(vectors_path),
            "vectors_sha256": _sha256(vectors_path),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
            "rows": int(groups.size),
            "episodes": int(np.unique(groups).size),
            "latent_dim": dim,
            "basis_artifacts": [str(path) for path in basis_paths],
            "basis_sha256": [_sha256(path) for path in basis_paths],
        },
        "protocol": {
            "external_targets": "encoded future embeddings from observed-action public PushT windows",
            "displacements": "target(second horizon) - target(first horizon)",
            "horizon_pairs": [list(pair) for pair in HORIZON_PAIRS],
            "k_values": ks,
            "random_subspace_repeats": int(random_repeats),
            "bootstrap_repeats": int(bootstrap_repeats),
            "bootstrap_unit": "episode",
            "basis_fit": "source-only frozen latent exports; no external refitting",
            "controls": ["covariance_basis", "dimension-matched random subspaces", "residual-sensitive bases when available"],
        },
        "records": source_records,
        "summary": summary,
        "claim_matrix": [
            {
                "claim": "A source-fitted dynamics basis captures external encoded latent displacement across multiple horizon spans.",
                "evidence": "dynamics-basis projection share exceeds covariance and random controls across all source splits and horizon pairs",
                "verdict": "supported as conditional latent-trajectory transport" if all(
                    summary[f"dynamics_basis/h{a}_to_h{b}/k{claim_k}"]["delta_vs_covariance_positive_all_sources"]
                    and summary[f"dynamics_basis/h{a}_to_h{b}/k{claim_k}"]["delta_vs_random_positive_all_sources"]
                    for a, b in HORIZON_PAIRS
                ) else "partial or not supported",
            },
            {
                "claim": "The transport is normalization-invariant.",
                "evidence": "this audit uses the already encoded external vectors and does not vary action normalization",
                "verdict": "not tested; do not claim normalization invariance",
            },
            {
                "claim": "The transported subspace is a physical or causal dynamics mechanism.",
                "evidence": "the target is a frozen model embedding displacement rather than a physical counterfactual",
                "verdict": "not supported by this diagnostic",
            },
        ],
        "claim_boundary": [
            "The basis is fit on source frozen latents and transported without refitting to external trajectories.",
            "External displacement is an encoded latent target difference, not a decoded physical state displacement.",
            "Positive projection-energy contrast establishes conditional latent-trajectory transport, not causal control, online planning, task success, or retraining gain.",
            "Action-normalization robustness is addressed separately by the cross-normalization audit and is not inferred here.",
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
        "# External multi-horizon dynamics-basis transport",
        "",
        "Source-fitted bases are transferred without refitting to encoded future embeddings from external public PushT windows. Each displacement is the difference between two encoded future horizons; values are episode-cluster summaries over independent source basis artifacts.",
        "",
        "| basis | displacement | k | projection share | vs covariance | vs random | all source signs positive |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for key, value in report["summary"].items():
        basis, pair, k_text = key.split("/")
        k = int(k_text[1:])
        lines.append(
            f"| {basis} | {pair} | {k} | {value['projection_share_mean_across_sources']:.6f} "
            f"| {value['delta_vs_covariance_mean_across_sources']:+.6f} "
            f"| {value['delta_vs_random_mean_across_sources']:+.6f} "
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
    parser.add_argument("--basis", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--random-repeats", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    run_report(
        vectors_path=args.vectors,
        metadata_path=args.metadata,
        basis_paths=args.basis,
        output=args.output,
        ks=args.ks,
        random_repeats=args.random_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
