"""Unify the TwoRoom failure and intervention evidence.

This report is descriptive and deliberately keeps the panels separate.  It
does not pool roots from different protocols into a causal estimate.  Its job
is to make the current diagnosis auditable: planner reachability ceiling,
action-ranking failure, offline-to-online transfer, and RGB fallback effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _success(payload: dict[str, Any], key: str) -> int:
    return int(sum(int(value) for value in payload[key]["episode_successes"]))


def _paired_binary(left: list[int], right: list[int]) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("paired binary vectors must have equal length")
    delta = [int(a) - int(b) for a, b in zip(left, right)]
    return {
        "left_wins": int(sum(value > 0 for value in delta)),
        "right_wins": int(sum(value < 0 for value in delta)),
        "ties": int(sum(value == 0 for value in delta)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def summarize(
    *, oracle: Path, sequence_ranker: Path, offline_ranker: Path,
    hybrid_panels: list[Path],
) -> dict[str, Any]:
    oracle_data = _load(oracle)
    sequence_data = _load(sequence_ranker)
    offline_data = _load(offline_ranker)
    if len(hybrid_panels) < 1:
        raise ValueError("at least one hybrid panel is required")

    oracle_metrics = oracle_data["metrics"]
    sequence_metrics = sequence_data["summary"]
    offline_summary = offline_data["summary"]
    panel_rows = []
    aggregate_frozen: list[int] = []
    aggregate_reactive: list[int] = []
    aggregate_hybrid: list[int] = []
    for panel_path in hybrid_panels:
        panel = _load(panel_path)
        frozen = [int(x) for x in panel["frozen_lewm"]["episode_successes"]]
        reactive = [int(x) for x in panel["rgb_reactive"]["episode_successes"]]
        hybrid = [int(x) for x in panel["hybrid_rgb_fallback"]["episode_successes"]]
        if not (len(frozen) == len(reactive) == len(hybrid)):
            raise ValueError(f"unpaired policy vectors in {panel_path}")
        aggregate_frozen.extend(frozen)
        aggregate_reactive.extend(reactive)
        aggregate_hybrid.extend(hybrid)
        panel_rows.append({
            "artifact": str(panel_path),
            "roots": len(frozen),
            "frozen_successes": int(sum(frozen)),
            "reactive_successes": int(sum(reactive)),
            "hybrid_successes": int(sum(hybrid)),
            "reactive_vs_frozen": _paired_binary(reactive, frozen),
            "hybrid_vs_frozen": _paired_binary(hybrid, frozen),
            "hybrid_vs_reactive": _paired_binary(hybrid, reactive),
            "hybrid_gate": _json_safe(panel["hybrid_rgb_fallback"].get("risk_gate", {})),
        })

    aggregate = {
        "roots": len(aggregate_frozen),
        "frozen_successes": int(sum(aggregate_frozen)),
        "reactive_successes": int(sum(aggregate_reactive)),
        "hybrid_successes": int(sum(aggregate_hybrid)),
        "frozen_rate": float(sum(aggregate_frozen) / len(aggregate_frozen)),
        "reactive_rate": float(sum(aggregate_reactive) / len(aggregate_reactive)),
        "hybrid_rate": float(sum(aggregate_hybrid) / len(aggregate_hybrid)),
        "hybrid_vs_frozen": _paired_binary(aggregate_hybrid, aggregate_frozen),
        "hybrid_vs_reactive": _paired_binary(aggregate_hybrid, aggregate_reactive),
    }
    return {
        "schema": "tworoom_failure_decomposition.v0.1",
        "planner_reachability_ceiling": {
            "oracle_roots": int(oracle_metrics["root_count"]),
            "oracle_available_roots": int(oracle_metrics["oracle_available_roots"]),
            "oracle_available_rate": float(oracle_metrics["oracle_available_rate"]),
            "zero_success_roots": int(oracle_metrics["zero_success_roots"]),
            "interpretation": "privileged candidate reachability ceiling; not learned-model evidence",
        },
        "learned_action_ranking": {
            "mean_spearman": float(sequence_metrics["mean_spearman"]),
            "top1_accuracy": float(sequence_metrics["top1_accuracy"]),
            "top4_accuracy": float(sequence_metrics["top4_accuracy"]),
            "mean_regret": float(sequence_metrics["mean_regret"]),
            "true_rank_of_predicted_best": float(sequence_metrics["mean_true_rank_of_predicted_best"]),
        },
        "offline_to_online_gap": {
            "offline_held_out_raw_regret": float(offline_summary["held_out_raw"]["mean_regret"]),
            "offline_held_out_ranker_regret": float(offline_summary["held_out_latent_action_ranker"]["mean_regret"]),
            "offline_regret_delta_ranker_minus_raw": float(offline_summary["held_out_delta_ranker_minus_raw_regret"]),
            "claim_boundary": "offline ranker gain does not establish online receding-horizon gain",
        },
        "online_intervention": {"panels": panel_rows, "aggregate": aggregate},
        "diagnosis": [
            "planner reachability is an independent ceiling on a subset of roots",
            "on reachable candidate panels, frozen learned action ranking is weak",
            "offline action readout improvement does not reliably transfer to online execution",
            "RGB re-observation fallback improves part of the online consequence and is close to the reactive ceiling, but does not establish superiority over it",
        ],
        "claim_boundary": (
            "descriptive multi-panel failure decomposition; no universal world-model, "
            "causal-physics, or adaptive-CEM certification claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--sequence-ranker", type=Path, required=True)
    parser.add_argument("--offline-ranker", type=Path, required=True)
    parser.add_argument("--hybrid-panel", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        oracle=args.oracle,
        sequence_ranker=args.sequence_ranker,
        offline_ranker=args.offline_ranker,
        hybrid_panels=args.hybrid_panel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
