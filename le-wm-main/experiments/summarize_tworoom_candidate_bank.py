"""Summarize matched standard-CEM versus candidate-bank TwoRoom panels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "live_tworoom_planning.v1":
        raise ValueError(f"unexpected schema: {path}")
    return payload


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
    parser.add_argument("--standard", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate-bank", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261062)
    args = parser.parse_args()
    if len(args.standard) != len(args.candidate_bank):
        raise ValueError("standard and candidate-bank panel counts must match")
    deltas: list[int] = []
    panels: list[dict[str, Any]] = []
    for standard_path, bank_path in zip(args.standard, args.candidate_bank):
        standard = _read(standard_path)
        bank = _read(bank_path)
        standard_seeds = np.asarray(standard["paired"]["seed_order"], dtype=np.int64)
        bank_seeds = np.asarray(bank["paired"]["seed_order"], dtype=np.int64)
        if not np.array_equal(standard_seeds, bank_seeds):
            raise ValueError(f"seed mismatch: {standard_path} vs {bank_path}")
        standard_success = np.asarray(standard["frozen_lewm"]["episode_successes"], dtype=np.int64)
        bank_success = np.asarray(bank["frozen_lewm"]["episode_successes"], dtype=np.int64)
        if standard_success.shape != bank_success.shape:
            raise ValueError("success vectors do not match")
        pair = bank_success - standard_success
        deltas.extend(pair.tolist())
        panels.append({
            "standard": str(standard_path),
            "candidate_bank": str(bank_path),
            "n": int(len(pair)),
            "standard_successes": int(standard_success.sum()),
            "candidate_bank_successes": int(bank_success.sum()),
            "candidate_bank_minus_standard": pair.tolist(),
        })
    values = np.asarray(deltas, dtype=np.int64)
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    payload = {
        "schema": "tworoom_candidate_bank_intervention.v1",
        "protocol": {
            "comparison": "same frozen checkpoint, same seed, same CEM budget; only fixed candidate bank differs",
            "candidate_bank": "zero, axis-aligned magnitudes 0.25/0.50/0.75, and four diagonals",
            "runtime_input": "rendered RGB current frame and rendered RGB goal only",
            "runtime_state_access": False,
            "device": "cuda",
            "panel_selection": "all supplied matched panels; no planning-outcome filtering",
        },
        "panels": panels,
        "paired": {
            "n": int(len(values)),
            "candidate_bank_minus_standard": values.tolist(),
            "wins": wins,
            "losses": losses,
            "ties": int(np.sum(values == 0)),
            "success_rate_delta": float(values.mean()),
            "bootstrap_ci95": _bootstrap(values, args.seed),
            "one_sided_candidate_bank_better_p": _sign_test(wins, losses),
        },
        "claim_boundary": "Matched TwoRoom candidate-support intervention; not a universal transfer certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
