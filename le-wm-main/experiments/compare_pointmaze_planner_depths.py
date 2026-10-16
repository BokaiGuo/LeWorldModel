"""Paired comparison of privileged PointMaze planner-depth ceilings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {int(row["episode_id"]): row for row in payload.get("rows", [])}
    if not rows or len(rows) != len(payload.get("rows", [])):
        raise ValueError(f"{path}: rows must have unique episode_id values")
    methods = {str(row.get("method")) for row in rows.values()}
    if methods != {"perfect_dynamics_mpc"}:
        raise ValueError(f"{path}: expected only perfect_dynamics_mpc, got {methods}")
    return payload, rows


def _bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indexes].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def compare(
    shallow_path: Path,
    deep_path: Path,
    *,
    draws: int = 10000,
    seed: int = 20261016,
) -> dict[str, Any]:
    shallow_payload, shallow = _load_rows(shallow_path)
    deep_payload, deep = _load_rows(deep_path)
    episodes = sorted(set(shallow) & set(deep))
    if set(shallow) != set(deep) or not episodes:
        raise ValueError("planner reports must contain exactly the same episodes")
    shallow_success = np.asarray([float(shallow[e]["success"]) for e in episodes])
    deep_success = np.asarray([float(deep[e]["success"]) for e in episodes])
    shallow_distance = np.asarray([float(shallow[e]["final_distance"]) for e in episodes])
    deep_distance = np.asarray([float(deep[e]["final_distance"]) for e in episodes])
    shallow_blocks = np.asarray([float(shallow[e]["blocks_executed"]) for e in episodes])
    deep_blocks = np.asarray([float(deep[e]["blocks_executed"]) for e in episodes])
    return {
        "schema": "pointmaze_privileged_depth_comparison.v1",
        "protocol": {
            "shallow_report": str(shallow_path.resolve()),
            "deep_report": str(deep_path.resolve()),
            "episode_paired": True,
            "episodes": episodes,
            "bootstrap_draws": draws,
            "seed": seed,
            "shallow_action_bank": shallow_payload["protocol"]["action_bank"],
            "deep_action_bank": deep_payload["protocol"]["action_bank"],
        },
        "paired": {
            "deep_minus_shallow_success": _bootstrap(
                deep_success - shallow_success, draws=draws, seed=seed
            ),
            "shallow_minus_deep_final_distance": _bootstrap(
                shallow_distance - deep_distance, draws=draws, seed=seed + 1
            ),
            "shallow_minus_deep_blocks": _bootstrap(
                shallow_blocks - deep_blocks, draws=draws, seed=seed + 2
            ),
            "discordant_success": {
                "deep_only": int(np.sum((deep_success == 1) & (shallow_success == 0))),
                "shallow_only": int(np.sum((deep_success == 0) & (shallow_success == 1))),
            },
        },
        "claim_boundary": (
            "privileged exact-dynamics reachability ceiling only; not RGB-only or "
            "frozen-world-model planning evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    report = compare(args.shallow, args.deep, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
