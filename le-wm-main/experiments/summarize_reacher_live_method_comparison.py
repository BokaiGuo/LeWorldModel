"""Compare live Reacher planners on the same paired seed panel."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in {
        "live_reacher_planning.v1",
        "live_reacher_sequence_candidate_planning.v1",
    }:
        raise ValueError(f"unexpected schema in {path}")
    protocol = payload["protocol"]
    arm = payload["frozen_lewm"]
    seeds = np.asarray(arm["seeds"], dtype=np.int64)
    successes = np.asarray(arm["episode_successes"], dtype=np.int64)
    random = payload["random"]
    random_seeds = np.asarray(random["seeds"], dtype=np.int64)
    random_successes = np.asarray(random["episode_successes"], dtype=np.int64)
    if not np.array_equal(seeds, random_seeds) or successes.shape != seeds.shape:
        raise ValueError(f"invalid paired arms in {path}")
    if random_successes.shape != seeds.shape or not np.all(np.isin(successes, (0, 1))):
        raise ValueError(f"invalid success vectors in {path}")
    return {"path": str(path), "protocol": protocol, "seeds": seeds, "successes": successes, "random": random_successes}


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261650)
    args = parser.parse_args()
    baseline = _read(args.baseline)
    methods = [_read(path) for path in args.methods]
    fixed = ("checkpoint", "checkpoint_frozen", "device", "horizon", "receding_horizon", "action_block", "cem", "runtime_input", "runtime_state_access", "same_seed_pairing", "target_delta")
    for method in methods:
        for key in fixed:
            if method["protocol"].get(key) != baseline["protocol"].get(key):
                raise ValueError(f"protocol mismatch for {key}: {method['path']}")
        if not np.array_equal(method["seeds"], baseline["seeds"]):
            raise ValueError(f"seed panel mismatch: {method['path']}")
        if not np.array_equal(method["random"], baseline["random"]):
            raise ValueError(f"random control mismatch: {method['path']}")

    comparisons = []
    for method in methods:
        delta = method["successes"] - baseline["successes"]
        wins = int(np.sum(delta > 0))
        losses = int(np.sum(delta < 0))
        comparisons.append({
            "artifact": method["path"],
            "candidate_bank": method["protocol"].get("candidate_bank", "raw_gaussian"),
            "baseline_success_rate": float(baseline["successes"].mean()),
            "method_success_rate": float(method["successes"].mean()),
            "paired_delta": delta.tolist(),
            "wins": wins,
            "losses": losses,
            "ties": int(np.sum(delta == 0)),
            "success_rate_delta": float(delta.mean()),
            "bootstrap_ci95": _bootstrap(delta, args.bootstrap_seed),
            "one_sided_exact_sign_test_p": _sign_test(wins, losses),
        })
    payload = {
        "schema": "live_reacher_method_comparison.v1",
        "protocol": {
            "baseline": str(args.baseline),
            "methods": [str(path) for path in args.methods],
            "same_seed_panel": True,
            "episode_count": int(len(baseline["seeds"])),
            "seed_order": baseline["seeds"].tolist(),
            "planning_outcome_used_for_selection": False,
        },
        "baseline": {
            "success_rate": float(baseline["successes"].mean()),
            "random_success_rate": float(baseline["random"].mean()),
        },
        "comparisons": comparisons,
        "claim_boundary": "Paired live Reacher RGB method comparison only; no universal transfer claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
