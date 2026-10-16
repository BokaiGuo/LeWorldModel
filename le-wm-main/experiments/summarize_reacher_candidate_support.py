"""Decompose Reacher candidate-bank gains into support and ranking effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in {
        "reacher_shared_candidate_ranking.v1",
        "reacher_sequence_candidate_ranking.v1",
    }:
        raise ValueError(f"unsupported schema in {path}")
    rows = payload.get("roots") or payload.get("rows")
    if not rows:
        raise ValueError(f"missing root rows in {path}")
    return {"path": str(path), "payload": payload, "rows": rows}


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261550)
    args = parser.parse_args()
    constant = _read(args.constant)
    sequence = _read(args.sequence)
    c_protocol = constant["payload"]["protocol"]
    s_protocol = sequence["payload"]["protocol"]
    for key in ("target_delta", "roots", "history_len", "replay_steps", "device"):
        if c_protocol.get(key) != s_protocol.get(key):
            raise ValueError(f"protocol mismatch for {key}")
    c_rows = {int(row["seed"]): row for row in constant["rows"]}
    s_rows = {int(row["seed"]): row for row in sequence["rows"]}
    if set(c_rows) != set(s_rows):
        raise ValueError("root seed sets differ")

    seeds = sorted(c_rows)
    constant_oracle = np.asarray([
        min(c_rows[seed]["realized_cost"]) for seed in seeds
    ], dtype=np.float64)
    sequence_oracle = np.asarray([
        s_rows[seed]["oracle_cost"] for seed in seeds
    ], dtype=np.float64)
    constant_selected = np.asarray([
        c_rows[seed]["multistep_task_error"] for seed in seeds
    ], dtype=np.float64)
    sequence_selected = np.asarray([
        s_rows[seed]["multistep_task_error"] for seed in seeds
    ], dtype=np.float64)
    constant_regret = np.asarray([
        c_rows[seed]["planning_regret"] for seed in seeds
    ], dtype=np.float64)
    sequence_regret = np.asarray([
        s_rows[seed]["planning_regret"] for seed in seeds
    ], dtype=np.float64)

    support_gain = constant_oracle - sequence_oracle
    selected_gain = constant_selected - sequence_selected
    regret_gain = constant_regret - sequence_regret
    output = {
        "schema": "reacher_candidate_support_comparison.v1",
        "protocol": {
            "constant_artifact": str(args.constant),
            "sequence_artifact": str(args.sequence),
            "same_root_seeds": True,
            "target_delta": s_protocol["target_delta"],
            "roots": len(seeds),
            "planning_outcome_used_for_selection": False,
        },
        "summary": {
            "constant_candidate_count": int(c_protocol["candidate_count"]),
            "sequence_candidate_count": int(s_protocol["candidate_count"]),
            "constant_oracle_error": float(constant_oracle.mean()),
            "sequence_oracle_error": float(sequence_oracle.mean()),
            "oracle_support_gain_constant_minus_sequence": float(support_gain.mean()),
            "oracle_support_gain_ci95": _bootstrap(support_gain, args.bootstrap_seed),
            "constant_selected_error": float(constant_selected.mean()),
            "sequence_selected_error": float(sequence_selected.mean()),
            "selected_error_gain_constant_minus_sequence": float(selected_gain.mean()),
            "selected_error_gain_ci95": _bootstrap(selected_gain, args.bootstrap_seed + 1),
            "constant_regret": float(constant_regret.mean()),
            "sequence_regret": float(sequence_regret.mean()),
            "regret_gain_constant_minus_sequence": float(regret_gain.mean()),
            "regret_gain_ci95": _bootstrap(regret_gain, args.bootstrap_seed + 2),
            "selected_sequence_wins": int(np.sum(selected_gain > 0.0)),
            "selected_sequence_losses": int(np.sum(selected_gain < 0.0)),
            "selected_sequence_ties": int(np.sum(np.isclose(selected_gain, 0.0))),
        },
        "claim_boundary": "Action-support decomposition only; does not establish improved latent dynamics or online planning.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
