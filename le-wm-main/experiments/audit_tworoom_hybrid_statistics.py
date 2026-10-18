"""Paired uncertainty audit for frozen, reactive, and hybrid TwoRoom arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _paired_delta(payload: dict[str, Any], left: str, right: str) -> np.ndarray:
    left_values = np.asarray(payload[left]["episode_successes"], dtype=np.float64)
    right_values = np.asarray(payload[right]["episode_successes"], dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("paired arms have different root vectors")
    return left_values - right_values


def _sign_flip_pvalue(delta: np.ndarray) -> float | None:
    nonzero = delta[delta != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return None
    wins = int(np.sum(nonzero > 0.0))
    tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1))
    return float(min(1.0, 2.0 * tail / (2.0 ** n)))


def _bootstrap_ci(delta: np.ndarray, *, seed: int, draws: int = 20000) -> list[float]:
    if delta.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(draws, delta.size))
    means = delta[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summarize(delta: np.ndarray, *, seed: int) -> dict[str, Any]:
    return {
        "roots": int(delta.size),
        "mean_delta": float(delta.mean()) if delta.size else None,
        "wins": int(np.sum(delta > 0.0)),
        "losses": int(np.sum(delta < 0.0)),
        "ties": int(np.sum(delta == 0.0)),
        "non_tie_roots": int(np.sum(delta != 0.0)),
        "exact_two_sided_sign_flip_p": _sign_flip_pvalue(delta),
        "paired_bootstrap_ci95": _bootstrap_ci(delta, seed=seed),
    }


def audit(panels: list[Path]) -> dict[str, Any]:
    if not panels:
        raise ValueError("at least one panel is required")
    payloads = [_load(path) for path in panels]
    deltas = {
        "hybrid_minus_frozen": np.concatenate([
            _paired_delta(payload, "hybrid_rgb_fallback", "frozen_lewm")
            for payload in payloads
        ]),
        "reactive_minus_frozen": np.concatenate([
            _paired_delta(payload, "rgb_reactive", "frozen_lewm")
            for payload in payloads
        ]),
        "hybrid_minus_reactive": np.concatenate([
            _paired_delta(payload, "hybrid_rgb_fallback", "rgb_reactive")
            for payload in payloads
        ]),
    }
    result = {
        "schema": "tworoom_hybrid_paired_statistics.v0.1",
        "protocol": {
            "panel_count": len(panels),
            "paired_root_vectors": True,
            "bootstrap_draws": 20000,
            "bootstrap_seed": 20261018,
            "sign_flip_null": "independent symmetric signs on non-tie paired differences",
            "test_outcomes_used_for_gate_selection": False,
        },
        "comparisons": {
            name: _summarize(delta, seed=20261018 + index)
            for index, (name, delta) in enumerate(deltas.items())
        },
        "interpretation": (
            "The hybrid gain over frozen is a paired consequence estimate. "
            "The hybrid-versus-reactive comparison is retained as a ceiling comparison; "
            "no superiority claim is authorized by this audit."
        ),
        "claim_boundary": (
            "predeclared paired uncertainty audit for these TwoRoom panels; "
            "not a universal cross-environment result or adaptive-CEM certificate"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
