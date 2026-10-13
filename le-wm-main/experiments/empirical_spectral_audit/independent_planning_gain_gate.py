"""Pre-registered independent planning-consequence gate.

The gate consumes paired fresh-episode losses from a frozen baseline and a
frozen diagnostic controller.  It does not choose a metric, weight, snapshot,
or environment from the outcome episodes.  Positive gain means
``baseline_loss - diagnostic_loss > 0``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_PROTOCOL_FLAGS = (
    "metric_selection_frozen",
    "selector_weights_frozen",
    "test_planning_used_for_selection",
    "runtime_state_access",
    "privileged_successor_verification",
    "fresh_planning_episodes",
    "planner_frozen",
    "independent_outcome_split",
)


def _paired_test(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "p_one_sided": None, "n_pairs": 0}
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, values.size))
    null_means = signs @ values / values.size
    p_value = float((1.0 + np.sum(null_means >= observed - 1e-15)) / (draws + 1.0))
    bootstrap = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_one_sided": p_value,
        "n_pairs": int(values.size),
    }


def audit_independent_planning_gain(
    records: list[dict[str, Any]],
    *,
    min_gain: float = 0.0,
    alpha: float = 0.05,
    min_pairs: int = 10,
    min_environments: int = 1,
    bootstrap_seed: int = 20261013,
) -> dict[str, Any]:
    """Audit the independent planning gain contract and paired consequence."""
    if not records:
        raise ValueError("records must not be empty")
    if not np.isfinite(float(min_gain)) or not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("min_gain must be finite and alpha must lie in (0,1)")
    if int(min_pairs) < 1 or int(min_environments) < 1:
        raise ValueError("min_pairs and min_environments must be positive")
    errors: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        protocol = record.get("protocol", {})
        missing = [flag for flag in REQUIRED_PROTOCOL_FLAGS if flag not in protocol]
        if missing:
            errors.append({"index": index, "reason": "missing_protocol_flags", "fields": missing})
            continue
        failed = [flag for flag in REQUIRED_PROTOCOL_FLAGS if protocol.get(flag) is not False and flag in {
            "test_planning_used_for_selection", "runtime_state_access", "privileged_successor_verification"
        }]
        failed += [flag for flag in REQUIRED_PROTOCOL_FLAGS if protocol.get(flag) is not True and flag not in {
            "test_planning_used_for_selection", "runtime_state_access", "privileged_successor_verification"
        }]
        environment = str(record.get("environment", protocol.get("environment", "unknown")))
        episode_id = str(record.get("episode_id", index))
        key = (environment, episode_id)
        if failed:
            errors.append({"index": index, "reason": "protocol_gate_failed", "fields": sorted(set(failed))})
            continue
        try:
            baseline = float(record["baseline_loss"])
            diagnostic = float(record["diagnostic_loss"])
        except (KeyError, TypeError, ValueError):
            errors.append({"index": index, "reason": "missing_or_invalid_losses"})
            continue
        if not np.isfinite(baseline) or not np.isfinite(diagnostic):
            errors.append({"index": index, "reason": "nonfinite_loss"})
            continue
        if key in seen:
            errors.append({"index": index, "reason": "duplicate_environment_episode", "key": list(key)})
            continue
        seen.add(key)
        valid.append({
            "environment": environment,
            "episode_id": episode_id,
            "baseline_loss": baseline,
            "diagnostic_loss": diagnostic,
            "gain": baseline - diagnostic,
        })
    by_environment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_environment[record["environment"]].append(record)
    gains = np.asarray([record["gain"] for record in valid], dtype=np.float64)
    paired = _paired_test(gains, bootstrap_seed)
    environments = sorted(by_environment)
    per_environment = {}
    for index, (environment, rows) in enumerate(sorted(by_environment.items())):
        local = _paired_test(np.asarray([row["gain"] for row in rows]), bootstrap_seed + index + 1)
        per_environment[environment] = {**local, "n_pairs": len(rows)}
    protocol_passed = not errors
    statistical_gate = bool(
        paired["n_pairs"] >= int(min_pairs)
        and len(environments) >= int(min_environments)
        and paired["mean"] is not None
        and paired["mean"] >= float(min_gain)
        and paired["ci95_low"] is not None
        and paired["ci95_low"] >= float(min_gain)
        and paired["p_one_sided"] is not None
        and paired["p_one_sided"] < float(alpha)
    )
    return {
        "schema": "independent_planning_gain.v1",
        "status": "independent_planning_gain_passed" if protocol_passed and statistical_gate else "gated_independent_planning_gain",
        "protocol_gate": protocol_passed,
        "statistical_gate": statistical_gate,
        "positive_gain": bool(protocol_passed and statistical_gate),
        "estimand": "baseline held-out loss minus diagnostic held-out loss",
        "thresholds": {
            "min_gain": float(min_gain), "alpha": float(alpha),
            "min_pairs": int(min_pairs), "min_environments": int(min_environments),
        },
        "paired_inference": paired,
        "environment_count": len(environments),
        "environments": environments,
        "per_environment": per_environment,
        "valid_pair_count": len(valid),
        "invalid_records": errors,
        "claim_boundary": (
            "This gate accepts only a pre-frozen, paired, independent planning "
            "consequence. It does not prove universal visual planning or causal "
            "latent dynamics."
        ),
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError("input must be a JSON list or an object with a records list")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--min-environments", type=int, default=1)
    args = parser.parse_args()
    report = audit_independent_planning_gain(
        _load_records(args.input), min_gain=args.min_gain, alpha=args.alpha,
        min_pairs=args.min_pairs, min_environments=args.min_environments,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "paired_inference": report["paired_inference"]}, indent=2))


if __name__ == "__main__":
    main()
