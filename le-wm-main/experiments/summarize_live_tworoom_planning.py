"""Aggregate fixed-protocol live TwoRoom planning panels.

This utility is deliberately outcome-agnostic: it reads complete panel
artifacts, verifies that CEM and pairing metadata agree, and computes the
predeclared seed-level planning consequence without selecting panels by
their outcome.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap requires at least two paired episodes")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "seed": int(seed),
        "draws": int(draws),
    }


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
    non_ties = wins + losses
    if non_ties == 0:
        return None
    # One-sided exact sign test under P(win)=P(loss)=1/2.
    return float(sum(math.comb(non_ties, k) for k in range(wins, non_ties + 1)) / 2**non_ties)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "live_tworoom_planning.v1":
        raise ValueError(f"unexpected schema in {path}")
    paired = payload.get("paired")
    if not paired:
        raise ValueError(f"missing paired summary in {path}")
    delta = np.asarray(paired["delta_lewm_minus_random"], dtype=np.int64)
    seeds = np.asarray(paired["seed_order"], dtype=np.int64)
    if delta.ndim != 1 or seeds.shape != delta.shape or len(np.unique(seeds)) != len(seeds):
        raise ValueError(f"invalid seed/delta pairing in {path}")
    random = payload.get("random", {})
    lewm = payload.get("frozen_lewm", {})
    random_seeds = np.asarray(random.get("seeds", []), dtype=np.int64)
    lewm_seeds = np.asarray(lewm.get("seeds", []), dtype=np.int64)
    random_success = np.asarray(random.get("episode_successes", []), dtype=np.int64)
    lewm_success = np.asarray(lewm.get("episode_successes", []), dtype=np.int64)
    if not np.array_equal(seeds, random_seeds) or not np.array_equal(seeds, lewm_seeds):
        raise ValueError(f"arm seeds are not canonicalized identically in {path}")
    if random_success.shape != delta.shape or lewm_success.shape != delta.shape:
        raise ValueError(f"arm success vectors do not match paired deltas in {path}")
    protocol = payload["protocol"]
    return {
        "path": str(path),
        "payload": payload,
        "delta": delta,
        "seeds": seeds,
        "random_success": random_success,
        "lewm_success": lewm_success,
        "protocol": protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261040)
    args = parser.parse_args()

    panels = [_read(path) for path in args.inputs]
    reference = panels[0]["protocol"]
    fixed_keys = ("checkpoint", "checkpoint_frozen", "device", "horizon", "receding_horizon", "action_block", "cem", "runtime_input", "runtime_state_access", "same_seed_pairing")
    for panel in panels:
        for key in fixed_keys:
            if panel["protocol"].get(key) != reference.get(key):
                raise ValueError(f"protocol mismatch for {key}: {panel['path']}")
        if panel["protocol"].get("checkpoint_frozen") is not True:
            raise ValueError(f"checkpoint is not declared frozen: {panel['path']}")
        if panel["protocol"].get("same_seed_pairing") is not True:
            raise ValueError(f"panel is not declared paired: {panel['path']}")

    deltas = np.concatenate([panel["delta"] for panel in panels])
    random_success = np.concatenate([panel["random_success"] for panel in panels])
    lewm_success = np.concatenate([panel["lewm_success"] for panel in panels])
    if not np.all(np.isin(deltas, (-1, 0, 1))):
        raise ValueError("planning success deltas must be binary paired differences")
    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))
    payload = {
        "schema": "live_tworoom_planning_aggregate.v1",
        "protocol": {
            **{key: reference[key] for key in fixed_keys},
            "panel_count": len(panels),
            "episode_count": int(deltas.size),
            "panel_artifacts": [panel["path"] for panel in panels],
            "panel_selection": "all supplied preregistered panels; no planning-outcome filtering",
        },
        "panel_summaries": [
            {
                "artifact": panel["path"],
                "n": int(panel["delta"].size),
                "wins": int(np.sum(panel["delta"] > 0)),
                "losses": int(np.sum(panel["delta"] < 0)),
                "ties": int(np.sum(panel["delta"] == 0)),
                "delta_mean": float(panel["delta"].mean()),
                "random_success_rate": float(panel["random_success"].mean()),
                "lewm_success_rate": float(panel["lewm_success"].mean()),
            }
            for panel in panels
        ],
        "paired": {
            "delta_lewm_minus_random": deltas.tolist(),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "success_rate_delta": float(deltas.mean()),
            "random_success_rate": float(random_success.mean()),
            "lewm_success_rate": float(lewm_success.mean()),
            "bootstrap": _bootstrap(deltas, args.bootstrap_seed),
            "one_sided_exact_sign_test_p": _sign_test_pvalue(wins, losses),
        },
        "claim_boundary": "Multi-panel live TwoRoom consequence only; not a universal world-model or cross-environment certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
