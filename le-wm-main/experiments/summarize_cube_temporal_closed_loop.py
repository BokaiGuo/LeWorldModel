"""Aggregate paired Cube temporal closed-loop panels without changing the protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "cube_temporal_closed_loop.v1":
        raise ValueError(f"unexpected schema in {path}")
    protocol = payload["protocol"]
    seeds = np.asarray(payload["paired"]["seed_order"], dtype=np.int64)
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError(f"duplicate test seeds in {path}")

    def by_seed(arm: str) -> dict[int, dict[str, Any]]:
        rows = payload[arm]["episodes"]
        indexed = {int(row["seed"]): row for row in rows}
        if set(indexed) != set(seeds.tolist()):
            raise ValueError(f"seed mismatch for {arm} in {path}")
        return indexed

    static = by_seed("static")
    temporal = by_seed("temporal")
    gate = by_seed("temporal_support_gate")
    feedback = by_seed("temporal_pixel_feedback") if "temporal_pixel_feedback" in payload else None
    feedback_local = by_seed("temporal_pixel_feedback_local") if "temporal_pixel_feedback_local" in payload else None
    temporal_delta = np.asarray(
        [temporal[int(seed)]["final_goal_distance_m"] - static[int(seed)]["final_goal_distance_m"] for seed in seeds],
        dtype=np.float64,
    )
    gate_delta = np.asarray(
        [gate[int(seed)]["final_goal_distance_m"] - static[int(seed)]["final_goal_distance_m"] for seed in seeds],
        dtype=np.float64,
    )
    feedback_delta = None
    if feedback is not None:
        feedback_delta = np.asarray(
            [feedback[int(seed)]["final_goal_distance_m"] - static[int(seed)]["final_goal_distance_m"] for seed in seeds],
            dtype=np.float64,
        )
    feedback_local_delta = None
    if feedback_local is not None:
        feedback_local_delta = np.asarray(
            [feedback_local[int(seed)]["final_goal_distance_m"] - static[int(seed)]["final_goal_distance_m"] for seed in seeds],
            dtype=np.float64,
        )
    triggered = np.asarray(
        [int(gate[int(seed)].get("support_fallback_count", 0)) > 0 for seed in seeds],
        dtype=bool,
    )
    return {
        "path": str(path),
        "protocol": protocol,
        "seeds": seeds,
        "temporal_delta": temporal_delta,
        "gate_delta": gate_delta,
        "feedback_delta": feedback_delta,
        "feedback_local_delta": feedback_local_delta,
        "triggered": triggered,
    }


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> list[float] | None:
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def _sign_test(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test after removing ties."""
    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    wins = int(np.sum(values < 0.0))
    losses = int(np.sum(values > 0.0))
    ties = int(np.sum(values == 0.0))
    return {
        "n": int(len(values)),
        "mean_delta_m": float(values.mean()) if len(values) else None,
        "median_delta_m": float(np.median(values)) if len(values) else None,
        "wins_improvement": wins,
        "losses_deterioration": losses,
        "ties": ties,
        "non_tie_improvement_rate": float(wins / (wins + losses)) if wins + losses else None,
        "bootstrap_ci95_mean_delta_m": _bootstrap(values, seed),
        "two_sided_exact_sign_test_p": _sign_test(wins, losses),
    }


def _protocol_signature(protocol: dict[str, Any]) -> tuple[Any, ...]:
    return (
        protocol.get("environment"),
        protocol.get("checkpoint"),
        protocol.get("checkpoint_frozen"),
        protocol.get("fixed_prelude"),
        protocol.get("support_gate"),
        protocol.get("support_threshold_m"),
        tuple(tuple(float(x) for x in row) for row in protocol.get("candidate_bank", [])),
        protocol.get("runtime_input"),
        protocol.get("runtime_policy_state_access"),
        protocol.get("pixel_feedback"),
        protocol.get("pixel_feedback_threshold"),
        protocol.get("pixel_feedback_mode", "persistent"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20262190)
    args = parser.parse_args()

    panels = [_load(path) for path in args.input]
    signature = _protocol_signature(panels[0]["protocol"])
    for panel in panels[1:]:
        if _protocol_signature(panel["protocol"]) != signature:
            raise ValueError(f"protocol mismatch: {panel['path']}")
    seeds = np.concatenate([panel["seeds"] for panel in panels])
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("test seed overlap across panels; cannot call panels independent")

    temporal = np.concatenate([panel["temporal_delta"] for panel in panels])
    gate = np.concatenate([panel["gate_delta"] for panel in panels])
    triggered = np.concatenate([panel["triggered"] for panel in panels])
    panel_reports = []
    for index, panel in enumerate(panels):
        panel_report = {
            "artifact": panel["path"],
            "test_seeds": panel["seeds"].tolist(),
            "n": int(len(panel["seeds"])),
            "temporal_minus_static": _summary(panel["temporal_delta"], args.bootstrap_seed + index),
            "support_gate_minus_static": _summary(panel["gate_delta"], args.bootstrap_seed + 100 + index),
            "support_triggered_episodes": int(panel["triggered"].sum()),
            "support_trigger_rate": float(panel["triggered"].mean()),
        }
        if panel["feedback_delta"] is not None:
            panel_report["pixel_feedback_minus_static"] = _summary(
                panel["feedback_delta"], args.bootstrap_seed + 200 + index,
            )
        if panel["feedback_local_delta"] is not None:
            panel_report["pixel_feedback_local_minus_static"] = _summary(
                panel["feedback_local_delta"], args.bootstrap_seed + 300 + index,
            )
        panel_reports.append(panel_report)

    triggered_gate = gate[triggered]
    untriggered_gate = gate[~triggered]
    pooled = {
        "temporal_minus_static": _summary(temporal, args.bootstrap_seed),
        "support_gate_minus_static": _summary(gate, args.bootstrap_seed + 1),
        "support_triggered_episodes": int(triggered.sum()),
        "support_trigger_rate": float(triggered.mean()),
        "support_triggered_gate_minus_static": _summary(triggered_gate, args.bootstrap_seed + 2),
        "support_untriggered_gate_minus_static": _summary(untriggered_gate, args.bootstrap_seed + 3),
    }
    feedback_panels = [panel for panel in panels if panel["feedback_delta"] is not None]
    if feedback_panels:
        if len(feedback_panels) != len(panels):
            raise ValueError("cannot pool pixel-feedback arm with panels that do not contain it")
        feedback = np.concatenate([panel["feedback_delta"] for panel in feedback_panels])
        pooled["pixel_feedback_minus_static"] = _summary(feedback, args.bootstrap_seed + 4)
    feedback_local_panels = [panel for panel in panels if panel["feedback_local_delta"] is not None]
    if feedback_local_panels:
        feedback_local = np.concatenate([panel["feedback_local_delta"] for panel in feedback_local_panels])
        pooled["pixel_feedback_local_minus_static"] = _summary(feedback_local, args.bootstrap_seed + 5)
        pooled["pixel_feedback_local_panel_count"] = len(feedback_local_panels)
    report = {
        "schema": "cube_temporal_closed_loop.aggregate.v1",
        "protocol": {
            "input_artifacts": [panel["path"] for panel in panels],
            "independent_test_seed_panels": True,
            "test_episode_count": int(len(seeds)),
            "seed_order": seeds.tolist(),
            "bootstrap_draws": 20000,
            "bootstrap_seed": int(args.bootstrap_seed),
            "planning_outcome_used_for_selection": False,
        },
        "panels": panel_reports,
        "pooled": pooled,
        "claim_boundary": (
            "Pooled paired RGB-only Cube consequence for a frozen temporal ranker "
            "and a fixed fail-closed support exploration rule. This is a scoped "
            "action-support result; it does not establish universal transfer, "
            "causal latent dynamics, or stable pose closure."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["pooled"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
