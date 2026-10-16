"""Summarize matched standard versus alternative-utility TwoRoom panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "live_tworoom_planning.v1":
        raise ValueError(f"unexpected schema in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", nargs="+", type=Path, required=True)
    parser.add_argument("--alternative", nargs="+", type=Path, required=True)
    parser.add_argument("--alternative-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.standard) != len(args.alternative):
        raise ValueError("standard and alternative panel counts must match")
    deltas: list[int] = []
    panels: list[dict] = []
    for standard_path, alternative_path in zip(args.standard, args.alternative):
        standard = _read(standard_path)
        alternative = _read(alternative_path)
        standard_seeds = np.asarray(standard["paired"]["seed_order"], dtype=np.int64)
        alternative_seeds = np.asarray(alternative["paired"]["seed_order"], dtype=np.int64)
        if not np.array_equal(standard_seeds, alternative_seeds):
            raise ValueError(f"seed mismatch: {standard_path} vs {alternative_path}")
        standard_success = np.asarray(standard["frozen_lewm"]["episode_successes"], dtype=np.int64)
        alternative_success = np.asarray(alternative["frozen_lewm"]["episode_successes"], dtype=np.int64)
        delta = alternative_success - standard_success
        deltas.extend(delta.tolist())
        panels.append({
            "standard": str(standard_path),
            "alternative": str(alternative_path),
            "n": int(len(delta)),
            "standard_successes": int(standard_success.sum()),
            "alternative_successes": int(alternative_success.sum()),
            "alternative_minus_standard": delta.tolist(),
        })
    values = np.asarray(deltas, dtype=np.int64)
    payload = {
        "schema": "tworoom_utility_intervention.v1",
        "protocol": {
            "comparison": "same frozen checkpoint, same seed, same CEM budget; only terminal utility differs",
            "standard": "full-dimensional Euclidean terminal latent MSE",
            "alternative": args.alternative_name,
            "runtime_input": "rendered RGB current frame and rendered RGB goal only",
            "runtime_state_access": False,
            "device": "cuda",
        },
        "panels": panels,
        "paired": {
            "n": int(len(values)),
            "alternative_minus_standard": values.tolist(),
            "wins": int(np.sum(values > 0)),
            "losses": int(np.sum(values < 0)),
            "ties": int(np.sum(values == 0)),
            "success_rate_delta": float(values.mean()),
        },
        "claim_boundary": "Matched TwoRoom utility intervention; not a universal transfer certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
