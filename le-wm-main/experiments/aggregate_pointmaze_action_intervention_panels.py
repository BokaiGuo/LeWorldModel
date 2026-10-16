"""Pool matched PointMaze action-intervention summaries across holdout panels.

Pooling is inverse-variance neutral and descriptive: each held-out episode
gets equal weight through the panel episode count. Split-level signs are kept
so a pooled gain cannot hide heterogeneity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pointmaze_action_intervention_summary.v1":
        raise ValueError(f"{path}: unexpected summary schema")
    if payload["protocol"].get("device") != "cuda":
        raise ValueError(f"{path}: expected a CUDA artifact")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = [_read(path) for path in args.panel]
    if len(panels) < 2:
        raise ValueError("at least two independently named holdout panels are required")

    reference = panels[0]["protocol"]
    for index, panel in enumerate(panels[1:], start=1):
        protocol = panel["protocol"]
        if protocol["checkpoint_sha256"] != reference["checkpoint_sha256"]:
            raise ValueError(f"panel {index}: checkpoint mismatch")
        if protocol["device"] != "cuda":
            raise ValueError(f"panel {index}: device mismatch")
        if set(panel["latent_prediction_mse"]) != {"observed", "zero", "episode_shuffle"}:
            raise ValueError(f"panel {index}: incomplete arm set")

    total_episodes = sum(int(panel["protocol"]["holdout_episode_count"]) for panel in panels)
    pooled_mse: dict[str, dict[str, float]] = {}
    pooled_deltas: dict[str, dict[str, float]] = {
        "observed_minus_zero_mse": {},
        "observed_minus_episode_shuffle_mse": {},
    }
    split_rows = []
    for horizon in ("1", "2", "4"):
        pooled_mse[horizon] = {}
        for arm in ("observed", "zero", "episode_shuffle"):
            numerator = sum(
                int(panel["protocol"]["holdout_episode_count"])
                * float(panel["latent_prediction_mse"][arm][horizon])
                for panel in panels
            )
            pooled_mse[horizon][arm] = numerator / total_episodes
        pooled_deltas["observed_minus_zero_mse"][horizon] = (
            pooled_mse[horizon]["observed"] - pooled_mse[horizon]["zero"]
        )
        pooled_deltas["observed_minus_episode_shuffle_mse"][horizon] = (
            pooled_mse[horizon]["observed"] - pooled_mse[horizon]["episode_shuffle"]
        )

    for path, panel in zip(args.panel, panels):
        row = {
            "artifact": str(path),
            "split": panel["protocol"]["split"],
            "holdout_episode_count": int(panel["protocol"]["holdout_episode_count"]),
            "deltas": panel["paired_mse_deltas"],
        }
        split_rows.append(row)

    report = {
        "schema": "pointmaze_action_intervention_panel_aggregate.v1",
        "protocol": {
            "panel_count": len(panels),
            "pooled_holdout_episode_count": total_episodes,
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "device": "cuda",
            "weighting": "equal weight per held-out episode via panel-count weighted means",
        },
        "pooled_latent_prediction_mse": pooled_mse,
        "pooled_mse_deltas": pooled_deltas,
        "split_level_results": split_rows,
        "interpretation": {
            "negative_delta": "observed actions improve prediction over the named control",
            "heterogeneity_preserved": True,
            "scope": "PointMaze temporal/action intervention across two disjoint tail holdout panels",
        },
        "claim_boundary": "does not establish action identifiability, closed-loop planning, causal latent physics, or a universal commutator certificate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
