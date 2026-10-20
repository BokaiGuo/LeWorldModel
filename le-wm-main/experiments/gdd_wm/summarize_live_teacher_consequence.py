"""Summarize a paired frozen-baseline versus teacher-readout consequence.

The script consumes two reader-facing planning artifacts produced with the
same seed panel.  It never re-selects a controller and reports paired
uncertainty separately from the unpaired success rates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("frozen_lewm"), dict):
        raise ValueError(f"{path}: missing frozen_lewm result")
    return payload


def _bootstrap(values: np.ndarray, seed: int, draws: int) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(draws, values.size))]
    means = sample.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _mcnemar_exact(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    right_only = int(((left == 0) & (right == 1)).sum())
    left_only = int(((left == 1) & (right == 0)).sum())
    discordant = right_only + left_only
    if discordant == 0:
        return right_only, left_only, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(right_only, left_only) + 1))
    return right_only, left_only, float(min(1.0, 2.0 * tail * 0.5**discordant))


def summarize(baseline_path: Path, intervention_path: Path, *, seed: int, draws: int) -> dict:
    baseline = _load(baseline_path)
    intervention = _load(intervention_path)
    base = baseline["frozen_lewm"]
    candidate = intervention["frozen_lewm"]
    base_seeds = np.asarray(base["seeds"], dtype=np.int64)
    candidate_seeds = np.asarray(candidate["seeds"], dtype=np.int64)
    if not np.array_equal(base_seeds, candidate_seeds):
        raise ValueError("baseline and intervention seed panels differ")
    base_success = np.asarray(base["episode_successes"], dtype=np.int64)
    candidate_success = np.asarray(candidate["episode_successes"], dtype=np.int64)
    base_distance = np.asarray(base["final_qpos_distance"], dtype=np.float64)
    candidate_distance = np.asarray(candidate["final_qpos_distance"], dtype=np.float64)
    if not (base_success.shape == candidate_success.shape == base_distance.shape == candidate_distance.shape):
        raise ValueError("paired result arrays have inconsistent shapes")
    success_delta = candidate_success - base_success
    distance_delta = candidate_distance - base_distance
    right_only, left_only, mcnemar_p = _mcnemar_exact(base_success, candidate_success)
    result = {
        "schema": "gdd_wm_paired_live_consequence_summary.v0.1",
        "protocol": {
            "baseline": str(baseline_path.resolve()),
            "intervention": str(intervention_path.resolve()),
            "same_seed_panel": True,
            "bootstrap_seed": int(seed),
            "bootstrap_draws": int(draws),
            "outcome_used_for_selection": False,
        },
        "n": int(base_success.size),
        "success": {
            "baseline": int(base_success.sum()),
            "intervention": int(candidate_success.sum()),
            "paired_delta_count": int(success_delta.sum()),
            "paired_delta_rate": float(success_delta.mean()),
            "paired_delta_ci95": _bootstrap(success_delta.astype(np.float64), seed, draws),
            "intervention_only": right_only,
            "baseline_only": left_only,
            "mcnemar_exact_p": mcnemar_p,
        },
        "final_distance": {
            "baseline_mean": float(base_distance.mean()),
            "intervention_mean": float(candidate_distance.mean()),
            "paired_delta_mean": float(distance_delta.mean()),
            "paired_delta_ci95": _bootstrap(distance_delta, seed + 1, draws),
        },
        "runtime_diagnostics": candidate.get("runtime_diagnostics"),
        "claim_boundary": (
            "paired positive direction is not an established independent GDD-WM gain "
            "unless the success and distance intervals exclude zero on a predeclared panel"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--intervention", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--draws", type=int, default=200000)
    args = parser.parse_args()
    result = summarize(args.baseline, args.intervention, seed=args.seed, draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
