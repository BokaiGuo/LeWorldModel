"""Aggregate predeclared Reacher target-geometry robustness panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if not n:
        return None
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "live_reacher_planning.v1":
        raise ValueError(f"unexpected schema in {path}")
    protocol = payload["protocol"]
    delta = np.asarray(payload["paired"]["delta_lewm_minus_random"], dtype=np.int64)
    seeds = np.asarray(payload["paired"]["seed_order"], dtype=np.int64)
    random = payload["random"]
    lewm = payload["frozen_lewm"]
    if not np.array_equal(seeds, np.asarray(random["seeds"], dtype=np.int64)):
        raise ValueError(f"random seeds are not paired in {path}")
    if not np.array_equal(seeds, np.asarray(lewm["seeds"], dtype=np.int64)):
        raise ValueError(f"LeWM seeds are not paired in {path}")
    if delta.shape != seeds.shape:
        raise ValueError(f"invalid paired vector in {path}")
    return {
        "path": str(path),
        "target_delta": tuple(float(x) for x in protocol["target_delta"]),
        "delta": delta,
        "random": np.asarray(random["episode_successes"], dtype=np.int64),
        "lewm": np.asarray(lewm["episode_successes"], dtype=np.int64),
        "protocol": protocol,
    }


def _summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    delta = np.concatenate([row["delta"] for row in rows])
    random = np.concatenate([row["random"] for row in rows])
    lewm = np.concatenate([row["lewm"] for row in rows])
    wins = int((delta > 0).sum())
    losses = int((delta < 0).sum())
    return {
        "panel_artifacts": [row["path"] for row in rows],
        "episode_count": int(len(delta)),
        "wins": wins,
        "losses": losses,
        "ties": int((delta == 0).sum()),
        "random_success_rate": float(random.mean()),
        "lewm_success_rate": float(lewm.mean()),
        "success_rate_delta": float(delta.mean()),
        "bootstrap_ci95": _bootstrap(delta.astype(np.float64), seed),
        "one_sided_exact_sign_test_p": _sign_p(wins, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [_read(path) for path in args.inputs]
    reference = rows[0]["protocol"]
    fixed = ("checkpoint", "checkpoint_frozen", "device", "task", "horizon", "receding_horizon", "action_block", "cem", "runtime_input", "runtime_state_access", "same_seed_pairing")
    for row in rows:
        for key in fixed:
            if row["protocol"].get(key) != reference.get(key):
                raise ValueError(f"protocol mismatch for {key}: {row['path']}")
        if row["protocol"].get("checkpoint_frozen") is not True or row["protocol"].get("same_seed_pairing") is not True:
            raise ValueError(f"panel is not frozen and paired: {row['path']}")
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["target_delta"], []).append(row)
    per_condition = {
        ",".join(f"{value:.2f}" for value in target_delta): _summary(group, 20261140 + index)
        for index, (target_delta, group) in enumerate(sorted(groups.items()))
    }
    pooled = _summary(rows, 20261190)
    payload = {
        "schema": "live_reacher_target_geometry_aggregate.v1",
        "protocol": {
            **{key: reference[key] for key in fixed},
            "panel_selection": "all supplied predeclared target-geometry panels; no planning-outcome filtering",
            "target_conditions": [list(key) for key in sorted(groups)],
        },
        "per_condition": per_condition,
        "pooled": pooled,
        "claim_boundary": "Target-geometry robustness of live Reacher RGB planning only; not a universal world-model or cross-environment certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"per_condition": per_condition, "pooled": pooled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
