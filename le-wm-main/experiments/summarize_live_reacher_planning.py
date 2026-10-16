"""Aggregate fixed-protocol live Reacher planning panels."""

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
    return float(sum(math.comb(non_ties, k) for k in range(wins, non_ties + 1)) / 2**non_ties)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in {
        "live_reacher_planning.v1",
        "live_reacher_sequence_candidate_planning.v1",
    }:
        raise ValueError(f"unexpected schema in {path}")
    paired = payload.get("paired")
    if not paired:
        raise ValueError(f"missing paired summary in {path}")
    seeds = np.asarray(paired["seed_order"], dtype=np.int64)
    delta = np.asarray(paired["delta_lewm_minus_random"], dtype=np.int64)
    random = payload["random"]
    lewm = payload["frozen_lewm"]
    random_seeds = np.asarray(random["seeds"], dtype=np.int64)
    lewm_seeds = np.asarray(lewm["seeds"], dtype=np.int64)
    random_success = np.asarray(random["episode_successes"], dtype=np.int64)
    lewm_success = np.asarray(lewm["episode_successes"], dtype=np.int64)
    if not np.array_equal(seeds, random_seeds) or not np.array_equal(seeds, lewm_seeds):
        raise ValueError(f"seed pairing mismatch in {path}")
    if delta.shape != seeds.shape or random_success.shape != delta.shape or lewm_success.shape != delta.shape:
        raise ValueError(f"invalid paired vector shapes in {path}")
    protocol = payload["protocol"]
    return {
        "path": str(path),
        "delta": delta,
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
    fixed_keys = (
        "checkpoint", "checkpoint_frozen", "device", "horizon",
        "receding_horizon", "action_block", "cem", "runtime_input",
        "runtime_state_access", "same_seed_pairing",
    )
    optional_fixed_keys = (
        ("task", "goal_construction")
        if panels[0]["protocol"].get("goal_construction") is not None
        else ("target_delta", "candidate_bank", "candidate_bank_in_cem_budget")
    )
    fixed_keys = fixed_keys + optional_fixed_keys
    for panel in panels:
        protocol = panel["protocol"]
        for key in fixed_keys:
            if protocol.get(key) != reference.get(key):
                raise ValueError(f"protocol mismatch for {key}: {panel['path']}")
        if protocol.get("checkpoint_frozen") is not True or protocol.get("same_seed_pairing") is not True:
            raise ValueError(f"panel is not frozen and paired: {panel['path']}")
    deltas = np.concatenate([panel["delta"] for panel in panels])
    random_success = np.concatenate([panel["random_success"] for panel in panels])
    lewm_success = np.concatenate([panel["lewm_success"] for panel in panels])
    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))
    payload = {
        "schema": "live_reacher_planning_aggregate.v1",
        "protocol": {
            **{key: reference[key] for key in fixed_keys},
            "panel_count": len(panels),
            "episode_count": int(deltas.size),
            "panel_artifacts": [panel["path"] for panel in panels],
            "panel_selection": "all supplied fixed-seed panels; no planning-outcome filtering",
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
        "claim_boundary": "Multi-panel live Reacher RGB planning consequence only; not a universal world-model or cross-environment certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
