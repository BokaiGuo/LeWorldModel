"""Summarize protocol-corrected temporal audits without pooling rows.

The unit of replication is the episode split, not an individual temporal
window.  This report keeps per-split values and reports wins across declared
splits; it never turns windows from one episode into independent seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one audit path is required")
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        memory = report["memory_closure"]
        signals = report["signal_information"]["reports"]
        paired_markov = memory["1"]["paired_outcome_rmse"]
        paired_m4 = memory.get("4", {}).get("paired_outcome_rmse")
        rows.append({
            "source": str(path),
            "split_seed": report["split"].get("split_seed"),
            "fit_episode_count": len(report["split"]["fit_episode_ids"]),
            "outcome_episode_count": len(report["split"]["outcome_episode_ids"]),
            "paired_outcome_current_count": report["protocol"]["memory_comparison"]["paired_outcome_current_count"],
            "paired_markov_rmse": paired_markov,
            "paired_memory4_rmse": paired_m4,
            "paired_memory4_win": bool(paired_m4 is not None and paired_m4 < paired_markov),
            "delay_relative_rmse_reduction": signals["delay"]["relative_rmse_reduction"],
            "interaction_relative_rmse_reduction": signals["interaction"]["relative_rmse_reduction"],
            "time_frequency_relative_rmse_reduction": signals["time_frequency"]["relative_rmse_reduction"],
            "matched_delay_relative_rmse_reduction": signals["delay"].get("fit_only_rank_matched", {}).get("relative_rmse_reduction"),
            "matched_interaction_relative_rmse_reduction": signals["interaction"].get("fit_only_rank_matched", {}).get("relative_rmse_reduction"),
            "matched_time_frequency_relative_rmse_reduction": signals["time_frequency"].get("fit_only_rank_matched", {}).get("relative_rmse_reduction"),
            "delay_win": bool(signals["delay"]["lifted_beats_base"]),
            "interaction_win": bool(signals["interaction"]["lifted_beats_base"]),
            "time_frequency_win": bool(signals["time_frequency"]["lifted_beats_base"]),
            "matched_delay_win": signals["delay"].get("fit_only_rank_matched", {}).get("lifted_beats_base"),
            "matched_interaction_win": signals["interaction"].get("fit_only_rank_matched", {}).get("lifted_beats_base"),
            "matched_time_frequency_win": signals["time_frequency"].get("fit_only_rank_matched", {}).get("lifted_beats_base"),
        })

    def mean_std(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}

    return {
        "protocol": {
            "replicate_unit": "seeded episode split",
            "row_pooling": False,
            "memory_comparison": "same current transitions supported by max declared history",
            "legacy_unpaired_values": "retained in each source artifact but excluded from win counts",
        },
        "n_splits": len(rows),
        "per_split": rows,
        "wins": {
            "paired_memory4": int(sum(row["paired_memory4_win"] for row in rows)),
            "delay": int(sum(row["delay_win"] for row in rows)),
            "interaction": int(sum(row["interaction_win"] for row in rows)),
            "time_frequency": int(sum(row["time_frequency_win"] for row in rows)),
            "matched_delay": int(sum(bool(row["matched_delay_win"]) for row in rows if row["matched_delay_win"] is not None)),
            "matched_interaction": int(sum(bool(row["matched_interaction_win"]) for row in rows if row["matched_interaction_win"] is not None)),
            "matched_time_frequency": int(sum(bool(row["matched_time_frequency_win"]) for row in rows if row["matched_time_frequency_win"] is not None)),
        },
        "aggregates": {
            key: mean_std(key)
            for key in (
                "paired_markov_rmse",
                "paired_memory4_rmse",
                "delay_relative_rmse_reduction",
                "interaction_relative_rmse_reduction",
                "time_frequency_relative_rmse_reduction",
                "matched_delay_relative_rmse_reduction",
                "matched_interaction_relative_rmse_reduction",
                "matched_time_frequency_relative_rmse_reduction",
            )
            if any(row[key] is not None for row in rows)
        },
        "claim_boundary": (
            "Protocol-corrected PointMaze temporal diagnostic; split-level evidence only. "
            "It does not establish a universal memory law, causal identification, online planning, "
            "or a universal world model."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
