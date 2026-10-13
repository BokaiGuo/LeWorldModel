"""Summarize paired terminal replay diagnostics without pooling episodes as seeds.

The first input is the fixed baseline.  Every later input is compared to it
episode by episode, using the episode index as the pairing key.  The script is
diagnostic only: it never selects a method or hides a negative replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    episodes = payload.get("planning", {}).get("episodes", [])
    if not episodes:
        raise ValueError(f"no planning episodes in {path}")
    return payload


def _episode(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace", [])
    failures = [item for item in trace if float(item.get("terminal_pulse_failed", 0.0)) > 0.0]
    fallbacks = [item for item in trace if float(item.get("terminal_pulse_fallback_used", 0.0)) > 0.0]
    frozen = [item for item in trace if float(item.get("terminal_pulse_frozen", 0.0)) > 0.0]
    reacq = [item for item in trace if float(item.get("reacquisition_attempt", 0.0)) > 0.0]
    reacq_success = [item for item in trace if float(item.get("reacquisition_success", 0.0)) > 0.0]
    return {
        "episode": int(row["episode"]),
        "strict_success": bool(row["strict_success"]),
        "block_xy_distance": float(row["block_xy_distance"]),
        "angle_error": float(row["angle_error"]),
        "first_pulse_failure_step": int(failures[0]["step"]) if failures else None,
        "pulse_failure_count": len(failures),
        "pulse_fallback_count": len(fallbacks),
        "pulse_frozen_count": len(frozen),
        "reacquisition_attempts": len(reacq),
        "reacquisition_successes": len(reacq_success),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "strict_success_count": int(sum(row["strict_success"] for row in rows)),
        "strict_success_rate": float(np.mean([row["strict_success"] for row in rows])),
        "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in rows])),
        "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])),
        "pulse_failure_episodes": int(sum(row["pulse_failure_count"] > 0 for row in rows)),
        "pulse_fallback_episodes": int(sum(row["pulse_fallback_count"] > 0 for row in rows)),
        "pulse_frozen_episodes": int(sum(row["pulse_frozen_count"] > 0 for row in rows)),
        "reacquisition_attempts": int(sum(row["reacquisition_attempts"] for row in rows)),
        "reacquisition_successes": int(sum(row["reacquisition_successes"] for row in rows)),
    }


def parse_label(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [parse_label(value) for value in args.input]
    loaded = [(label, _load(path)) for label, path in inputs]
    summaries: dict[str, Any] = {}
    episode_rows: dict[str, list[dict[str, Any]]] = {}
    for label, payload in loaded:
        rows = [_episode(row) for row in payload["planning"]["episodes"]]
        episode_rows[label] = rows
        summaries[label] = _summary(rows)

    baseline_label = inputs[0][0]
    baseline = {row["episode"]: row for row in episode_rows[baseline_label]}
    paired: dict[str, Any] = {}
    for label, _path in inputs[1:]:
        rows = episode_rows[label]
        common = [row["episode"] for row in rows if row["episode"] in baseline]
        if not common:
            raise ValueError(f"no paired episode ids for {baseline_label} vs {label}")
        candidate = {row["episode"]: row for row in rows}
        xy_delta = [candidate[index]["block_xy_distance"] - baseline[index]["block_xy_distance"] for index in common]
        angle_delta = [candidate[index]["angle_error"] - baseline[index]["angle_error"] for index in common]
        success_delta = [int(candidate[index]["strict_success"]) - int(baseline[index]["strict_success"]) for index in common]
        paired[label] = {
            "baseline": baseline_label,
            "paired_episodes": len(common),
            "mean_xy_delta_candidate_minus_baseline": float(np.mean(xy_delta)),
            "mean_angle_delta_candidate_minus_baseline": float(np.mean(angle_delta)),
            "strict_success_delta_candidate_minus_baseline": int(sum(success_delta)),
            "episode_rows": [
                {"episode": index, "xy_delta": float(xy_delta[pos]), "angle_delta": float(angle_delta[pos]), "success_delta": int(success_delta[pos])}
                for pos, index in enumerate(common)
            ],
        }

    report = {
        "protocol": {
            "baseline": baseline_label,
            "input_labels": [label for label, _path in inputs],
            "pairing_unit": "episode id within the fixed replay seed",
            "selection": "none; diagnostic summary only",
        },
        "summaries": summaries,
        "paired": paired,
        "episode_rows": episode_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "terminal_replay_diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = ["# Terminal replay diagnostics", "", f"- Baseline: `{baseline_label}`.", ""]
    for label, summary in summaries.items():
        lines.append(
            f"- `{label}`: success {summary['strict_success_count']}/{summary['episodes']}, "
            f"XY {summary['mean_block_xy_distance']:.4g}, angle {summary['mean_angle_error']:.4g}, "
            f"pulse-failure episodes {summary['pulse_failure_episodes']}, "
            f"reacquisition {summary['reacquisition_attempts']}/{summary['reacquisition_successes']}."
        )
    for label, result in paired.items():
        lines.append(
            f"- Paired `{label}` minus `{result['baseline']}`: "
            f"XY {result['mean_xy_delta_candidate_minus_baseline']:.4g}, "
            f"angle {result['mean_angle_delta_candidate_minus_baseline']:.4g}, "
            f"success delta {result['strict_success_delta_candidate_minus_baseline']}."
        )
    (args.output / "terminal_replay_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "summaries": summaries, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
