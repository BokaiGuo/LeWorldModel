"""Paired adjudication of a Reacher capacity readout against frozen LeWM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired bootstrap requires at least two one-dimensional rows")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indexes].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "live_reacher_sequence_candidate_planning.v1":
        raise ValueError(f"{path}: unexpected schema")
    return payload


def compare(
    baseline_path: Path,
    readout_path: Path,
    *,
    draws: int = 20_000,
    seed: int = 20261016,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    readout = _load(readout_path)
    baseline_protocol = baseline["protocol"]
    readout_protocol = readout["protocol"]
    required_equal = (
        "checkpoint",
        "horizon",
        "receding_horizon",
        "action_block",
        "candidate_bank",
        "candidate_count",
        "final_action_selection",
        "target_delta",
    )
    mismatches = {
        key: [baseline_protocol.get(key), readout_protocol.get(key)]
        for key in required_equal
        if baseline_protocol.get(key) != readout_protocol.get(key)
    }
    if mismatches:
        raise ValueError(f"planning protocols are not matched: {mismatches}")
    if baseline_protocol.get("cost_metric") != "euclidean":
        raise ValueError("baseline must use euclidean cost")
    if readout_protocol.get("cost_metric") != "capacity_safe_readout":
        raise ValueError("candidate must use capacity_safe_readout")
    if readout_protocol.get("final_action_selection") != "best":
        raise ValueError("offline top-1 adjudication requires online best execution")

    baseline_rows = baseline["frozen_lewm"]
    readout_rows = readout["frozen_lewm"]
    if baseline_rows["seeds"] != readout_rows["seeds"]:
        raise ValueError("baseline and readout must contain the same ordered seeds")
    success_base = np.asarray(baseline_rows["episode_successes"], dtype=np.float64)
    success_readout = np.asarray(readout_rows["episode_successes"], dtype=np.float64)
    final_base = np.asarray(baseline_rows["final_qpos_distance"], dtype=np.float64)
    final_readout = np.asarray(readout_rows["final_qpos_distance"], dtype=np.float64)
    min_base = np.asarray(baseline_rows["min_qpos_distance"], dtype=np.float64)
    min_readout = np.asarray(readout_rows["min_qpos_distance"], dtype=np.float64)
    if not (
        len(success_base)
        == len(success_readout)
        == len(final_base)
        == len(final_readout)
        == len(min_base)
        == len(min_readout)
    ):
        raise ValueError("paired result arrays have different lengths")

    diagnostics = readout_rows.get("runtime_diagnostics", {})
    return {
        "schema": "reacher_capacity_readout_paired_comparison.v1",
        "protocol": {
            "baseline_report": str(baseline_path.resolve()),
            "readout_report": str(readout_path.resolve()),
            "episode_paired": True,
            "seeds": baseline_rows["seeds"],
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "final_action_selection": "best",
            "capacity_readout_artifact": readout_protocol.get("capacity_readout_artifact"),
        },
        "point_estimates": {
            "baseline_successes": int(np.sum(success_base)),
            "readout_successes": int(np.sum(success_readout)),
            "baseline_mean_final_qpos_distance": float(np.mean(final_base)),
            "readout_mean_final_qpos_distance": float(np.mean(final_readout)),
            "baseline_mean_min_qpos_distance": float(np.mean(min_base)),
            "readout_mean_min_qpos_distance": float(np.mean(min_readout)),
        },
        "paired_readout_minus_baseline": {
            "success": _bootstrap(success_readout - success_base, draws=draws, seed=seed),
            "final_qpos_distance": _bootstrap(
                final_readout - final_base, draws=draws, seed=seed + 1
            ),
            "min_qpos_distance": _bootstrap(
                min_readout - min_base, draws=draws, seed=seed + 2
            ),
            "success_discordance": {
                "readout_only_wins": int(np.sum((success_readout == 1) & (success_base == 0))),
                "baseline_only_wins": int(np.sum((success_readout == 0) & (success_base == 1))),
                "ties": int(np.sum(success_readout == success_base)),
            },
        },
        "runtime_diagnostics": {
            "mlp_gate_enabled": bool(diagnostics.get("mlp_no_dynamics_gate_enabled", False)),
            "active_rows": int(diagnostics.get("mlp_active_rows", 0)),
            "evaluated_rows": int(diagnostics.get("mlp_evaluated_rows", 0)),
            "abstained_low_support_rows": int(
                diagnostics.get("mlp_abstained_low_support_rows", 0)
            ),
            "min_support_fraction": float(
                diagnostics.get("mlp_min_support_fraction", 0.0)
            ),
            "proposal_support_rate": float(diagnostics.get("mlp_proposal_support_rate", 0.0)),
            "row_support_fraction_mean": float(
                diagnostics.get(
                    "mlp_row_support_fraction_mean",
                    diagnostics.get("mlp_proposal_support_rate", 0.0),
                )
            ),
            "row_support_fraction_min": float(
                diagnostics.get("mlp_row_support_fraction_min", 0.0)
            ),
            "row_support_fraction_max": float(
                diagnostics.get("mlp_row_support_fraction_max", 0.0)
            ),
            "argmin_changes": int(diagnostics.get("mlp_argmin_changes", 0)),
            "argmin_change_rate": float(
                diagnostics.get("mlp_argmin_changes", 0)
                / max(int(diagnostics.get("mlp_evaluated_rows", 0)), 1)
            ),
        },
        "claim_boundary": (
            "fresh paired Reacher online consequence for a phase-gated source-action "
            "readout; not independent predicted-dynamics, universal-model, or "
            "cross-environment evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    report = compare(args.baseline, args.readout, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
