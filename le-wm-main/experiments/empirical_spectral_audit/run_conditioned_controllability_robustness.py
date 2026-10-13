"""Robustness audit for state-conditioned action-to-dynamics readouts.

This is a frozen PushT latent diagnostic.  It evaluates whether the held-out
action-to-dynamics signal survives changes in sample cap and state-cluster
resolution, while comparing against a pooled action readout, a state-only
cell-mean predictor, and an action-permutation null.  It does not train LeWM
or provide trajectory, causal, control, or planning evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_metrics import cluster_states


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    centered = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    return float(1.0 - np.square(target - prediction).sum() / max(denominator, 1e-12))


def _split(indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(indices)
    cut = max(1, indices.size // 2)
    if cut >= indices.size:
        cut = indices.size - 1
    return indices[:cut], indices[cut:]


def _fit_predict(
    actions: np.ndarray,
    dynamics: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    intercept: bool,
) -> float:
    if intercept:
        train_x = np.concatenate(
            [np.ones((train.size, 1), dtype=np.float32), actions[train]], axis=1
        )
        test_x = np.concatenate(
            [np.ones((test.size, 1), dtype=np.float32), actions[test]], axis=1
        )
    else:
        train_x = actions[train]
        test_x = actions[test]
    coefficient, *_ = np.linalg.lstsq(train_x, dynamics[train], rcond=None)
    return _r2(dynamics[test], test_x @ coefficient)


def _cell_record(
    actions: np.ndarray,
    dynamics: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    cell_results: list[dict[str, float | int]] = []
    for label in sorted(np.unique(labels).tolist()):
        cell = np.flatnonzero(labels == label)
        if cell.size < 4:
            continue
        train, test = _split(cell, seed + 101 * (int(label) + 1))
        action_r2 = _fit_predict(actions, dynamics, train, test, intercept=False)
        intercept_r2 = _fit_predict(actions, dynamics, train, test, intercept=True)
        mean_prediction = np.broadcast_to(
            dynamics[train].mean(axis=0, keepdims=True), (test.size, dynamics.shape[1])
        )
        state_only_r2 = _r2(dynamics[test], mean_prediction)

        shuffled_actions = actions[cell].copy()
        np.random.default_rng(seed + 10_003 + int(label)).shuffle(shuffled_actions)
        shuffled_train = np.flatnonzero(np.isin(cell, train))
        shuffled_test = np.flatnonzero(np.isin(cell, test))
        null_coefficient, *_ = np.linalg.lstsq(
            shuffled_actions[shuffled_train], dynamics[cell][shuffled_train], rcond=None
        )
        permutation_r2 = _r2(
            dynamics[cell][shuffled_test],
            shuffled_actions[shuffled_test] @ null_coefficient,
        )
        cell_results.append(
            {
                "label": int(label),
                "n": int(cell.size),
                "test_n": int(test.size),
                "action_r2": float(action_r2),
                "intercept_action_r2": float(intercept_r2),
                "state_only_r2": float(state_only_r2),
                "permutation_r2": float(permutation_r2),
            }
        )

    weights = np.asarray([float(item["test_n"]) for item in cell_results])
    weights /= weights.sum()

    def weighted(name: str) -> float:
        return float(np.sum(weights * [float(item[name]) for item in cell_results]))

    action = weighted("action_r2")
    permutation = weighted("permutation_r2")
    return {
        "num_cells": len(cell_results),
        "min_cell_n": int(min(item["n"] for item in cell_results)),
        "min_test_n": int(min(item["test_n"] for item in cell_results)),
        "cell_action_r2_mean": float(np.mean([item["action_r2"] for item in cell_results])),
        "cell_action_r2_weighted": action,
        "cell_intercept_action_r2_weighted": weighted("intercept_action_r2"),
        "cell_state_only_r2_weighted": weighted("state_only_r2"),
        "cell_permutation_r2_weighted": permutation,
        "cell_action_minus_state_only": action - weighted("state_only_r2"),
        "cell_action_minus_permutation": action - permutation,
        "cell_results": cell_results,
    }


def _pooled_record(
    actions: np.ndarray, dynamics: np.ndarray, seed: int
) -> dict[str, float]:
    train, test = _split(np.arange(actions.shape[0]), seed + 900_001)
    return {
        "pooled_action_r2": _fit_predict(actions, dynamics, train, test, intercept=False),
        "pooled_intercept_action_r2": _fit_predict(actions, dynamics, train, test, intercept=True),
        "pooled_mean_r2": _r2(
            dynamics[test],
            np.broadcast_to(dynamics[train].mean(axis=0, keepdims=True), (test.size, dynamics.shape[1])),
        ),
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((int(record["sample_cap"]), int(record["clusters"])), []).append(record)
    summaries = []
    for (sample_cap, num_clusters), rows in sorted(grouped.items()):
        def values(path: tuple[str, str]) -> np.ndarray:
            return np.asarray([float(row[path[0]][path[1]]) for row in rows], dtype=np.float64)

        action = values(("conditioned", "cell_action_r2_weighted"))
        state_only = values(("conditioned", "cell_state_only_r2_weighted"))
        permutation = values(("conditioned", "cell_permutation_r2_weighted"))
        pooled = values(("pooled", "pooled_action_r2"))
        summaries.append(
            {
                "sample_cap": sample_cap,
                "clusters": num_clusters,
                "num_seeds": len(rows),
                "conditional_action_r2_mean": float(action.mean()),
                "conditional_action_r2_std": float(action.std(ddof=1)) if action.size > 1 else 0.0,
                "pooled_action_r2_mean": float(pooled.mean()),
                "state_only_r2_mean": float(state_only.mean()),
                "permutation_r2_mean": float(permutation.mean()),
                "action_minus_state_only_mean": float((action - state_only).mean()),
                "action_minus_permutation_mean": float((action - permutation).mean()),
                "all_seeds_action_above_state_only": bool(np.all(action > state_only)),
                "all_seeds_action_above_permutation": bool(np.all(action > permutation)),
            }
        )
    return summaries


def run_report(
    source: Path,
    output: Path,
    sample_caps: list[int],
    clusters: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        required = {"actions", "dynamics", "states"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in required}
    n = arrays["actions"].shape[0]
    if any(value.shape[0] != n for value in arrays.values()):
        raise ValueError("array row counts do not match")

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for cap in sample_caps:
            if cap <= 0 or cap > n:
                raise ValueError(f"sample cap {cap} must be in [1, {n}]")
            rows = np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))
            actions = arrays["actions"][rows]
            dynamics = arrays["dynamics"][rows]
            states = arrays["states"][rows]
            pooled = _pooled_record(actions, dynamics, seed)
            for num_clusters in clusters:
                labels = cluster_states(states, num_clusters, seed=seed)
                conditioned = _cell_record(actions, dynamics, labels, seed + num_clusters)
                record = {
                    "seed": int(seed),
                    "sample_cap": int(cap),
                    "clusters": int(num_clusters),
                    "pooled": pooled,
                    "conditioned": conditioned,
                }
                records.append(record)
                print(
                    f"seed={seed} cap={cap} clusters={num_clusters} "
                    f"conditional_r2={conditioned['cell_action_r2_weighted']:.5f} "
                    f"null={conditioned['cell_permutation_r2_weighted']:.5f}",
                    flush=True,
                )
    report = {
        "artifact": "conditioned_controllability_robustness",
        "status": "frozen_offline_boundary_evidence",
        "source": str(source),
        "protocol": {
            "sample_caps": [int(value) for value in sample_caps],
            "clusters": [int(value) for value in clusters],
            "seeds": [int(value) for value in seeds],
            "fit": "zero-intercept and intercept action-to-dynamics least squares",
            "split": "within-cell random half for conditioned models; random half for pooled models",
            "controls": ["pooled action", "state-only cell mean", "within-cell action permutation"],
            "target": "one-step frozen dynamics",
            "end_to_end_retraining": False,
            "rollout_and_planning_available": False,
        },
        "records": records,
        "aggregates": _aggregate(records),
        "claim_boundary": [
            "This is a frozen latent, state-conditioned offline readout audit.",
            "The action signal is interpreted only when it exceeds the state-only and permutation controls across the declared sweep.",
            "It does not establish causal controllability, online control, planning success, physical observability, or multi-step rollout quality.",
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
        "# Frozen PushT conditioned controllability robustness",
        "",
        "This sweep cross-fits action-to-dynamics readouts on frozen PushT latent rows. It varies sample cap and state-cluster resolution and compares pooled action, state-conditioned action, a state-only cell mean, and an action-permutation null.",
        "",
        "## Across-seed aggregates",
        "",
        "| rows | clusters | conditional action R2 (mean +/- SD) | pooled action R2 | state-only R2 | permutation R2 | action minus state-only | action minus null | all seeds pass both controls |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for summary in report["aggregates"]:
        passed = summary["all_seeds_action_above_state_only"] and summary["all_seeds_action_above_permutation"]
        lines.append(
            f"| {summary['sample_cap']} | {summary['clusters']} | "
            f"{summary['conditional_action_r2_mean']:.6f} +/- {summary['conditional_action_r2_std']:.6f} | "
            f"{summary['pooled_action_r2_mean']:.6f} | {summary['state_only_r2_mean']:.6f} | "
            f"{summary['permutation_r2_mean']:.6f} | {summary['action_minus_state_only_mean']:.6f} | "
            f"{summary['action_minus_permutation_mean']:.6f} | {'yes' if passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed records",
            "",
        "| seed | rows | clusters | pooled action R2 | conditional action R2 | state-only R2 | permutation R2 | conditional minus state-only | conditional minus null |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in report["records"]:
        conditioned = record["conditioned"]
        pooled = record["pooled"]
        lines.append(
            f"| {record['seed']} | {record['sample_cap']} | {record['clusters']} | "
            f"{pooled['pooled_action_r2']:.6f} | {conditioned['cell_action_r2_weighted']:.6f} | "
            f"{conditioned['cell_state_only_r2_weighted']:.6f} | {conditioned['cell_permutation_r2_weighted']:.6f} | "
            f"{conditioned['cell_action_minus_state_only']:.6f} | {conditioned['cell_action_minus_permutation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-caps", type=int, nargs="+", default=[20000, 100000, 200000])
    parser.add_argument("--clusters", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    args = parser.parse_args()
    run_report(args.source, args.output, args.sample_caps, args.clusters, args.seeds)
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
