"""Summarize a matched PointMaze action-intervention panel.

The report is deliberately descriptive: it compares observed actions with
zero-action and cross-episode-shuffled controls on the same held-out split.
It does not fit a planning-risk predictor or promote commutator values to a
transfer certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol", {}).get("device") != "cuda":
        raise ValueError(f"{path}: PointMaze intervention must be a CUDA artifact")
    return payload


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        str(h): float(payload["latent_prediction_error_by_horizon"][str(h)]["mean"])
        for h in (1, 2, 4)
    }


def _validate_match(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    ref = reference["protocol"]
    got = candidate["protocol"]
    for key in ("split", "holdout_episode_count", "episodes_used", "horizons", "checkpoint_sha256"):
        if ref.get(key) != got.get(key):
            raise ValueError(f"{label}: protocol mismatch in {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--episode-shuffle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed = _read(args.observed)
    zero = _read(args.zero)
    shuffled = _read(args.episode_shuffle)
    _validate_match(observed, zero, "zero")
    _validate_match(observed, shuffled, "episode_shuffle")
    if observed["protocol"].get("action_mode") != "observed":
        raise ValueError("observed input must declare action_mode=observed")
    if zero["protocol"].get("action_mode") != "zero":
        raise ValueError("zero input must declare action_mode=zero")
    if shuffled["protocol"].get("action_mode") != "episode_shuffle":
        raise ValueError("episode-shuffle input has the wrong action mode")

    arms = {
        "observed": _metrics(observed),
        "zero": _metrics(zero),
        "episode_shuffle": _metrics(shuffled),
    }
    deltas = {
        "observed_minus_zero_mse": {h: arms["observed"][h] - arms["zero"][h] for h in arms["observed"]},
        "observed_minus_episode_shuffle_mse": {
            h: arms["observed"][h] - arms["episode_shuffle"][h] for h in arms["observed"]
        },
    }
    report = {
        "schema": "pointmaze_action_intervention_summary.v1",
        "protocol": {
            "split": observed["protocol"]["split"],
            "holdout_episode_count": observed["protocol"]["holdout_episode_count"],
            "checkpoint_sha256": observed["protocol"]["checkpoint_sha256"],
            "device": "cuda",
            "arms": {
                "observed": str(args.observed),
                "zero": str(args.zero),
                "episode_shuffle": str(args.episode_shuffle),
            },
        },
        "latent_prediction_mse": arms,
        "paired_mse_deltas": deltas,
        "interpretation": {
            "observed_minus_zero": "negative means observed actions improve prediction over zero-action control",
            "observed_minus_episode_shuffle": "negative means aligned within-episode actions improve prediction over shuffled actions",
            "scope": "held-out PointMaze temporal/action intervention only",
        },
        "claim_boundary": "does not establish action identifiability, closed-loop planning, causal latent physics, or a universal commutator certificate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
