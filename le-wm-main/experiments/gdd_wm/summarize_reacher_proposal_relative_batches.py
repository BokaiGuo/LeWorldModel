"""Summarize root-paired Reacher proposal-relative planning batches."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    intervention = int(np.sum((a == 1) & (b == 0)))
    baseline = int(np.sum((a == 0) & (b == 1)))
    discordant = intervention + baseline
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, k) for k in range(min(intervention, baseline) + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "intervention_only": intervention,
        "baseline_only": baseline,
        "ties": int(len(a) - discordant),
        "exact_mcnemar_p": float(p_value),
    }


def _bootstrap(values: np.ndarray, *, seed: int, draws: int = 20_000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indexes].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_unit": "episode_within_disjoint_batch_pool",
    }


def summarize(pairs: list[tuple[Path, Path]], *, seed: int = 20261020) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError("at least two independent paired batches are required")
    all_teacher: list[int] = []
    all_baseline: list[int] = []
    all_distance: list[float] = []
    all_seeds: list[int] = []
    reports: list[dict[str, Any]] = []
    artifact_ids: set[str | None] = set()
    for index, (teacher_path, baseline_path) in enumerate(pairs):
        teacher = _read(teacher_path)
        baseline = _read(baseline_path)
        t = teacher["frozen_lewm"]
        b = baseline["frozen_lewm"]
        if t["seeds"] != b["seeds"]:
            raise ValueError(f"batch {index}: teacher/baseline seeds differ")
        seeds = [int(value) for value in t["seeds"]]
        if set(all_seeds).intersection(seeds):
            raise ValueError("paired batches must have disjoint seeds")
        all_seeds.extend(seeds)
        teacher_success = np.asarray(t["episode_successes"], dtype=np.float64)
        baseline_success = np.asarray(b["episode_successes"], dtype=np.float64)
        distance_delta = np.asarray(t["final_qpos_distance"], dtype=np.float64) - np.asarray(
            b["final_qpos_distance"], dtype=np.float64
        )
        all_teacher.extend(teacher_success.astype(int).tolist())
        all_baseline.extend(baseline_success.astype(int).tolist())
        all_distance.extend(distance_delta.tolist())
        artifact_ids.add(teacher["protocol"].get("capacity_readout_artifact"))
        reports.append({
            "batch_index": index,
            "teacher_report": str(teacher_path.resolve()),
            "baseline_report": str(baseline_path.resolve()),
            "n": len(seeds),
            "teacher_successes": int(teacher_success.sum()),
            "baseline_successes": int(baseline_success.sum()),
            "distance_delta": float(distance_delta.mean()),
        })
    if len(artifact_ids) != 1:
        raise ValueError("all batches must use the same capacity artifact")
    teacher_array = np.asarray(all_teacher, dtype=np.float64)
    baseline_array = np.asarray(all_baseline, dtype=np.float64)
    distance = np.asarray(all_distance, dtype=np.float64)
    return {
        "schema": "gdd_wm_reacher_proposal_relative_paired_aggregate.v1",
        "protocol": {
            "batch_count": len(pairs),
            "root_disjoint": True,
            "seeds": all_seeds,
            "capacity_readout_artifact": next(iter(artifact_ids)),
            "test_outcome_used_for_gate_tuning": False,
            "execution_mode": "mean",
        },
        "batches": reports,
        "pooled": {
            "n": len(all_seeds),
            "teacher_successes": int(teacher_array.sum()),
            "baseline_successes": int(baseline_array.sum()),
            "success_rate_delta": float((teacher_array - baseline_array).mean()),
            "success_discordance": _mcnemar(teacher_array, baseline_array),
            "final_qpos_distance_delta": _bootstrap(distance, seed=seed),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", metavar=("TEACHER", "BASELINE"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize([(Path(pair[0]), Path(pair[1])) for pair in args.pair])
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
