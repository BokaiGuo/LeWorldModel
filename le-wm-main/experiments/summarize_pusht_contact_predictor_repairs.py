"""Summarize independent teacher-forced PushT predictor-repair runs.

The summary preserves seed-level rows and deliberately does not pool episode
bootstrap intervals across runs.  It is intended as the reader-facing source
for the supplement table, not as evidence for the unmodified frozen model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="repair JSON; repeat for independent seeds")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run_report(inputs: list[Path]) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("provide at least two independent repair runs")
    rows = []
    for path in inputs:
        report = json.loads(path.read_text())
        seed = int(report["protocol"].get("seed", path.parent.name.rsplit("seed", 1)[-1].split("_", 1)[0]))
        for item in report["horizons"]:
            metrics = item["metrics"]
            rows.append(
                {
                    "seed": seed,
                    "horizon": int(item["horizon"]),
                    "base_error_plus": metrics["base_error_plus"],
                    "repaired_calibrated_error_plus": metrics["repaired_calibrated_error_plus"],
                    "paired_calibrated_gain_plus": metrics["paired_calibrated_gain_plus"],
                    "paired_native_gain_plus": metrics["paired_native_gain_plus"],
                    "base_latent_counterfactual_separation": metrics["base_latent_counterfactual_separation"],
                    "repaired_latent_counterfactual_separation": metrics["repaired_latent_counterfactual_separation"],
                    "physical_counterfactual_separation": metrics["physical_counterfactual_separation"],
                }
            )
    horizons = sorted({row["horizon"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    by_horizon = {}
    for horizon in horizons:
        subset = [row for row in rows if row["horizon"] == horizon]
        gains = [row["paired_calibrated_gain_plus"]["estimate"] for row in subset]
        native = [row["paired_native_gain_plus"]["estimate"] for row in subset]
        by_horizon[str(horizon)] = {
            "seed_count": len(subset),
            "point_gain_positive_all_seeds": all(value > 0 for value in gains),
            "native_gain_negative_all_seeds": all(value < 0 for value in native),
            "point_gain_by_seed": {str(row["seed"]): row["paired_calibrated_gain_plus"]["estimate"] for row in subset},
            "ci_low_by_seed": {str(row["seed"]): row["paired_calibrated_gain_plus"]["ci95_low"] for row in subset},
        }
    return {
        "protocol": {
            "independent_seeds": seeds,
            "seed_level_rows": len(rows),
            "aggregation": "no pooling; report each seed and episode-bootstrap interval separately",
            "comparison": "base predictor-native readout versus repaired predictor-specific calibrated readout",
        },
        "rows": rows,
        "by_horizon": by_horizon,
        "claim_boundary": (
            "Contact-rich teacher-forced repair is a targeted diagnostic for one published PushT checkpoint. "
            "It does not establish that the original frozen checkpoint had the repaired behavior, nor universal "
            "physical causality, online planning gain, or task success."
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_report([Path(value) for value in args.input])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_predictor_teacher_forced_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Contact predictor teacher-forced repair summary",
        "",
        "| seed | h | base + error | repaired calibrated + error | paired gain (+ is better) | paired gain 95% CI | native gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        gain = row["paired_calibrated_gain_plus"]
        lines.append(
            "| %d | %d | %.3f | %.3f | %.3f | [%.3f, %.3f] | %.3f |" % (
                row["seed"], row["horizon"],
                row["base_error_plus"]["estimate"],
                row["repaired_calibrated_error_plus"]["estimate"],
                gain["estimate"], gain["ci95_low"], gain["ci95_high"],
                row["paired_native_gain_plus"]["estimate"],
            )
        )
    lines += [
        "",
        "Rows remain separated by independent seed; confidence intervals are not pooled.",
        "Claim boundary: targeted repair diagnostic, not evidence for the unmodified frozen model's causal control or planning ability.",
        "",
    ]
    (output / "contact_predictor_teacher_forced_summary.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
