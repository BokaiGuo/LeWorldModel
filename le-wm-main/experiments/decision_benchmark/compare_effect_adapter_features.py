"""Compare frozen single-frame and causal-history task-effect adapters.

Both adapters are trained/evaluated on one immutable candidate bank.  The
comparison is over independent optimizer seeds; outcome roots are never used
to choose a feature family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap(values: np.ndarray, seed: int, samples: int = 5000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean(values[rng.integers(0, len(values), len(values))])
        for _ in range(samples)
    ])
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "seeds": int(len(values)),
    }


def compare(single_report: Path, history_report: Path, output: Path) -> dict[str, Any]:
    single = _load(single_report)
    history = _load(history_report)
    single_rows = {int(row["seed"]): row for row in single["reports"]}
    history_rows = {int(row["seed"]): row for row in history["reports"]}
    if set(single_rows) != set(history_rows):
        raise ValueError("single-frame and history reports have different seed sets")
    seeds = sorted(single_rows)
    regret_delta = np.asarray([
        history_rows[seed]["selected_regret"]["estimate"] - single_rows[seed]["selected_regret"]["estimate"]
        for seed in seeds
    ])
    success_delta = np.asarray([
        history_rows[seed]["selected_success"]["estimate"] - single_rows[seed]["selected_success"]["estimate"]
        for seed in seeds
    ])
    rows = [{
        "seed": seed,
        "single_frame_selected_regret": single_rows[seed]["selected_regret"],
        "history_selected_regret": history_rows[seed]["selected_regret"],
        "history_minus_single_regret": float(regret_delta[index]),
        "single_frame_selected_success": single_rows[seed]["selected_success"],
        "history_selected_success": history_rows[seed]["selected_success"],
        "history_minus_single_success": float(success_delta[index]),
    } for index, seed in enumerate(seeds)]
    report = {
        "schema": "paired_history_effect_adapter.v1",
        "protocol": {
            "pairing_unit": "adapter seed",
            "candidate_bank": "same history4-conditioned bank, same root split, same action sequences",
            "single_frame_report": str(single_report.resolve()),
            "history_report": str(history_report.resolve()),
            "outcome_used_for_feature_selection": False,
            "uncertainty": "seed bootstrap; outcome rows remain internal to each report's fixed evaluation",
        },
        "rows": rows,
        "summary": {
            "seeds": len(seeds),
            "single_frame_selected_regret": _bootstrap(np.asarray([single_rows[s]["selected_regret"]["estimate"] for s in seeds]), 20261021),
            "history_selected_regret": _bootstrap(np.asarray([history_rows[s]["selected_regret"]["estimate"] for s in seeds]), 20261022),
            "history_minus_single_regret": _bootstrap(regret_delta, 20261023),
            "single_frame_selected_success": _bootstrap(np.asarray([single_rows[s]["selected_success"]["estimate"] for s in seeds]), 20261024),
            "history_selected_success": _bootstrap(np.asarray([history_rows[s]["selected_success"]["estimate"] for s in seeds]), 20261025),
            "history_minus_single_success": _bootstrap(success_delta, 20261026),
        },
        "interpretation": "Negative history-minus-single regret is favorable; positive history-minus-single success is favorable.",
        "claim_boundary": "Held-out offline PointMaze action-effect consequence; not RGB online closed-loop planning or universal transfer.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-report", type=Path, required=True)
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.single_report, args.history_report, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
