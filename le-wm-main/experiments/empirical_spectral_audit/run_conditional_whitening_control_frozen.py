"""Test whether action-conditioned local geometry survives global whitening.

Global whitening makes the pooled latent covariance identity by construction.
This audit asks whether action-bin conditional covariance/operator pairs still
have non-trivial commutators after that transformation.  The result is a
frozen one-step diagnostic, not a retrained or causal world-model result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_whitening_invariance_frozen import (
        _commutator,
        _fit_operator,
        _finite,
        _stable_rank,
        _whitener,
    )
except ModuleNotFoundError:  # direct invocation from the script directory
    from run_whitening_invariance_frozen import (  # type: ignore
        _commutator,
        _fit_operator,
        _finite,
        _stable_rank,
        _whitener,
    )


def _labels(actions: np.ndarray, bins: int) -> np.ndarray:
    score = np.linalg.norm(actions, axis=1)
    thresholds = np.quantile(score, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(score, thresholds, right=False).astype(np.int64)


def _one_record(
    source: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    seed: int,
    bins: int,
    ridge: float,
) -> dict[str, Any]:
    order = np.random.default_rng(seed + 8001).permutation(source.shape[0])
    cut = source.shape[0] // 2
    train = order[:cut]
    train_source = source[train]
    train_target = target[train]
    train_actions = actions[train]
    global_covariance, global_operator = _fit_operator(train_source, train_target, ridge)
    whitening, coloring = _whitener(global_covariance)
    global_white_covariance = whitening @ global_covariance @ whitening.T
    global_white_operator = whitening @ global_operator @ coloring

    labels = _labels(train_actions, bins)
    local_original = []
    local_whitened = []
    sizes = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        local_covariance, local_operator = _fit_operator(
            train_source[indices], train_target[indices], ridge
        )
        local_original.append(_commutator(local_covariance, local_operator))
        local_white_covariance = whitening @ local_covariance @ whitening.T
        local_white_operator = whitening @ local_operator @ coloring
        local_whitened.append(_commutator(local_white_covariance, local_white_operator))
        sizes.append(int(indices.size))
    weights = np.asarray(sizes, dtype=np.float64)
    weights /= weights.sum()

    def weighted(records: list[dict[str, float]], key: str) -> float:
        return float(np.dot(weights, [record[key] for record in records]))

    global_original = _commutator(global_covariance, global_operator)
    global_whitened = _commutator(global_white_covariance, global_white_operator)
    return {
        "seed": int(seed),
        "num_samples": int(source.shape[0]),
        "bins": int(bins),
        "train_samples": int(train.size),
        "group_sizes": sizes,
        "global_original": global_original,
        "global_whitened": global_whitened,
        "local_original_weighted": {
            "normalized_frobenius": weighted(local_original, "normalized_frobenius"),
            "normalized_spectral": weighted(local_original, "normalized_spectral"),
        },
        "local_whitened_weighted": {
            "normalized_frobenius": weighted(local_whitened, "normalized_frobenius"),
            "normalized_spectral": weighted(local_whitened, "normalized_spectral"),
        },
        "local_original_stable_rank_mean": float(
            np.dot(weights, [
                _stable_rank(_fit_operator(train_source[np.flatnonzero(labels == label)], train_target[np.flatnonzero(labels == label)], ridge)[0])
                for label in sorted(np.unique(labels).tolist())
            ])
        ),
    }


def run_report(
    source: Path,
    output: Path,
    sample_caps: list[int],
    seeds: list[int],
    bins: list[int],
    ridge: float,
) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        required = {"latents", "predicted_dynamics", "actions"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        latents = np.asarray(archive["latents"], dtype=np.float64)
        predicted = np.asarray(archive["predicted_dynamics"], dtype=np.float64)
        actions = np.asarray(archive["actions"], dtype=np.float64)
    if latents.shape != predicted.shape or latents.shape[0] != actions.shape[0]:
        raise ValueError("source arrays have incompatible shapes")
    target = latents + predicted
    records = []
    for seed in seeds:
        for cap in sample_caps:
            rows = np.sort(np.random.default_rng(seed).choice(latents.shape[0], size=cap, replace=False))
            for num_bins in bins:
                record = _one_record(latents[rows], target[rows], actions[rows], seed, num_bins, ridge)
                records.append(record)
                print(
                    f"seed={seed} rows={cap} bins={num_bins} "
                    f"global_white={record['global_whitened']['normalized_frobenius']:.3g} "
                    f"local_white={record['local_whitened_weighted']['normalized_frobenius']:.6g}",
                    flush=True,
                )
    report = {
        "artifact": "conditional_geometry_after_global_whitening",
        "status": "frozen_offline_boundary_evidence",
        "source": str(source),
        "protocol": {
            "sample_caps": [int(value) for value in sample_caps],
            "seeds": [int(value) for value in seeds],
            "action_bins": [int(value) for value in bins],
            "fit": "centered ridge operators fit on a disjoint train half",
            "global_transform": "inverse square root of pooled train covariance",
            "target": "latents + predicted_dynamics",
            "rollout_and_planning_available": False,
        },
        "records": records,
        "claim_boundary": [
            "Global whitening removes pooled covariance anisotropy by construction.",
            "Non-zero action-conditioned commutators after whitening indicate that conditional geometry is not equivalent to the pooled global commutator.",
            "This is frozen one-step operator evidence; it does not establish retraining gains, causal action effects, rollout, or planning.",
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
        "# Conditional geometry after global whitening on frozen LeWM latents",
        "",
        "Global whitening collapses the pooled commutator. This audit checks whether action-conditioned local covariance/operator pairs retain non-zero commutators after the same coordinate transformation.",
        "",
        "| seed | rows | bins | global C before | global C after whitening | local C before | local C after whitening | local stable rank |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['seed']} | {record['num_samples']} | {record['bins']} | "
            f"{record['global_original']['normalized_frobenius']:.6g} | "
            f"{record['global_whitened']['normalized_frobenius']:.3g} | "
            f"{record['local_original_weighted']['normalized_frobenius']:.6g} | "
            f"{record['local_whitened_weighted']['normalized_frobenius']:.6g} | "
            f"{record['local_original_stable_rank_mean']:.3f} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-caps", type=int, nargs="+", default=[100000, 200000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--bins", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    run_report(args.source, args.output, args.sample_caps, args.seeds, args.bins, args.ridge)
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
