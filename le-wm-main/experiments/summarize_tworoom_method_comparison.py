"""Compare live TwoRoom planning methods on the exact same seed panel.

The comparator keeps the random control and seed order from each planning
artifact, then reports paired method-minus-baseline success differences.  It
does not select a method or a seed panel by outcome and is intentionally
limited to the supplied live TwoRoom consequence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "live_tworoom_planning.v1":
        raise ValueError(f"unexpected schema in {path}")
    protocol = payload.get("protocol", {})
    if protocol.get("checkpoint_frozen") is not True:
        raise ValueError(f"checkpoint is not frozen in {path}")
    if protocol.get("same_seed_pairing") is not True:
        raise ValueError(f"panel is not paired in {path}")
    arm = payload.get("frozen_lewm", {})
    seeds = np.asarray(arm.get("seeds", []), dtype=np.int64)
    successes = np.asarray(arm.get("episode_successes", []), dtype=np.int64)
    random = payload.get("random", {})
    random_seeds = np.asarray(random.get("seeds", []), dtype=np.int64)
    random_successes = np.asarray(random.get("episode_successes", []), dtype=np.int64)
    if seeds.ndim != 1 or len(seeds) < 2 or len(np.unique(seeds)) != len(seeds):
        raise ValueError(f"invalid seeds in {path}")
    if not np.array_equal(seeds, random_seeds):
        raise ValueError(f"random and planning seeds disagree in {path}")
    if successes.shape != seeds.shape or random_successes.shape != seeds.shape:
        raise ValueError(f"success vectors disagree in {path}")
    if not np.all(np.isin(successes, (0, 1))) or not np.all(np.isin(random_successes, (0, 1))):
        raise ValueError(f"success vectors must be binary in {path}")
    return {
        "path": str(path),
        "payload": payload,
        "protocol": protocol,
        "seeds": seeds,
        "successes": successes,
        "random_successes": random_successes,
    }


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "seed": int(seed),
        "draws": int(draws),
    }


def _sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261055)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    methods = [_load(path) for path in args.methods]
    fixed_keys = (
        "checkpoint", "checkpoint_frozen", "device", "horizon", "receding_horizon",
        "action_block", "cem", "candidate_bank", "runtime_input", "runtime_state_access",
        "same_seed_pairing",
    )
    for method in methods:
        for key in fixed_keys:
            if method["protocol"].get(key) != baseline["protocol"].get(key):
                raise ValueError(f"protocol mismatch for {key}: {method['path']}")
        if not np.array_equal(method["seeds"], baseline["seeds"]):
            raise ValueError(f"seed panel mismatch: {method['path']}")
        if not np.array_equal(method["random_successes"], baseline["random_successes"]):
            raise ValueError(f"random controls mismatch: {method['path']}")

    comparisons = []
    for method in methods:
        delta = method["successes"] - baseline["successes"]
        wins = int(np.sum(delta > 0))
        losses = int(np.sum(delta < 0))
        comparisons.append({
            "artifact": method["path"],
            "cost_metric": method["protocol"].get("cost_metric"),
            "ranker_artifact": method["protocol"].get("ranker_artifact"),
            "baseline_success_rate": float(baseline["successes"].mean()),
            "method_success_rate": float(method["successes"].mean()),
            "delta_method_minus_baseline": delta.tolist(),
            "wins": wins,
            "losses": losses,
            "ties": int(np.sum(delta == 0)),
            "paired_success_rate_delta": float(delta.mean()),
            "bootstrap": _bootstrap(delta, args.bootstrap_seed),
            "one_sided_exact_sign_test_p": _sign_test(wins, losses),
        })

    output = {
        "schema": "live_tworoom_method_comparison.v1",
        "protocol": {
            "baseline": str(args.baseline),
            "methods": [str(path) for path in args.methods],
            "same_seed_panel": True,
            "planning_outcome_used_for_selection": False,
            "episode_count": int(len(baseline["seeds"])),
            "seed_order": baseline["seeds"].tolist(),
        },
        "baseline": {
            "cost_metric": baseline["protocol"].get("cost_metric"),
            "success_rate": float(baseline["successes"].mean()),
            "random_success_rate": float(baseline["random_successes"].mean()),
        },
        "comparisons": comparisons,
        "claim_boundary": "Paired live TwoRoom method comparison only; no universal world-model or cross-environment certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
