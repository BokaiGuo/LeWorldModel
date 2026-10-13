"""Summarize the deep contact-predictor diagnostics without pooling seeds.

The report keeps seed-level bootstrap intervals separate and exposes the three
intervention arms, the observed-latent upper bound, and calibration-size
curves.  It is deliberately not a claim that the unmodified frozen model has
the repaired behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[int]]:
    if len(paths) < 2:
        raise ValueError("provide at least two independent seed artifacts")
    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    for path in paths:
        report = json.loads(path.read_text())
        protocol = report["protocol"]
        seed = int(protocol["seed"])
        seeds.append(seed)
        for item in report["horizons"]:
            h = int(item["horizon"])
            rows.append(
                {
                    "seed": seed,
                    "horizon": h,
                    "plus_gain": item["by_arm"]["plus"]["paired_calibrated_gain"],
                    "minus_gain": item["by_arm"]["minus"]["paired_calibrated_gain"],
                    "zero_gain": item["by_arm"]["zero"]["paired_calibrated_gain"],
                    "plus_upper_bound_error": item["metrics"][
                        "observed_latent_upper_bound_error_plus"
                    ],
                    "plus_base_error": item["metrics"]["base_error_plus"],
                    "plus_repaired_calibrated_error": item["metrics"][
                        "repaired_calibrated_error_plus"
                    ],
                    "physical_separation": item["metrics"][
                        "physical_counterfactual_separation"
                    ],
                    "base_latent_separation": item["metrics"][
                        "base_latent_counterfactual_separation"
                    ],
                    "repaired_latent_separation": item["metrics"][
                        "repaired_latent_counterfactual_separation"
                    ],
                    "calibration_ablation": item["calibration_ablation"],
                }
            )
    return rows, sorted(set(seeds))


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows, seeds = load_rows(paths)
    horizons = sorted({row["horizon"] for row in rows})
    by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        subset = [row for row in rows if row["horizon"] == horizon]
        by_horizon[str(horizon)] = {
            "seed_count": len(subset),
            "plus_point_gain_positive_all_seeds": all(
                row["plus_gain"]["estimate"] > 0 for row in subset
            ),
            "minus_point_gain_positive_all_seeds": all(
                row["minus_gain"]["estimate"] > 0 for row in subset
            ),
            "zero_point_gain_positive_all_seeds": all(
                row["zero_gain"]["estimate"] > 0 for row in subset
            ),
            "plus_ci_low_positive_all_seeds": all(
                row["plus_gain"]["ci95_low"] > 0 for row in subset
            ),
            "plus_gain_by_seed": {
                str(row["seed"]): row["plus_gain"]["estimate"] for row in subset
            },
            "plus_ci_low_by_seed": {
                str(row["seed"]): row["plus_gain"]["ci95_low"] for row in subset
            },
            "minus_gain_by_seed": {
                str(row["seed"]): row["minus_gain"]["estimate"] for row in subset
            },
            "zero_gain_by_seed": {
                str(row["seed"]): row["zero_gain"]["estimate"] for row in subset
            },
            "upper_bound_error_by_seed": {
                str(row["seed"]): row["plus_upper_bound_error"]["estimate"]
                for row in subset
            },
        }
    calibration: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["horizon"])
        calibration.setdefault(key, {})[str(row["seed"])] = row["calibration_ablation"]
    return {
        "protocol": {
            "seeds": seeds,
            "seed_level_rows": len(rows),
            "aggregation": "seed-level rows remain separate; no pooled bootstrap interval",
            "arms": ["plus", "minus", "zero"],
            "upper_bound": "decoder trained on observed target latents and evaluated on held-out target latents",
        },
        "rows": rows,
        "by_horizon": by_horizon,
        "calibration_ablation": calibration,
        "claim_boundary": (
            "Targeted predictor-side repair diagnostic for one published PushT checkpoint. "
            "It does not establish original frozen-checkpoint causality, universal physical "
            "counterfactual accuracy, online planning gain, or task success."
        ),
    }


def main() -> None:
    args = parse_args()
    report = summarize([Path(value) for value in args.input])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_predictor_deep_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Deep contact-predictor repair summary",
        "",
        "| seed | h | +a gain | -a gain | zero gain | +a upper-bound error | physical +/- separation |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| %d | %d | %.3f [% .3f, %.3f] | %.3f | %.3f | %.3f | %.3f |"
            % (
                row["seed"],
                row["horizon"],
                row["plus_gain"]["estimate"],
                row["plus_gain"]["ci95_low"],
                row["plus_gain"]["ci95_high"],
                row["minus_gain"]["estimate"],
                row["zero_gain"]["estimate"],
                row["plus_upper_bound_error"]["estimate"],
                row["physical_separation"]["estimate"],
            )
        )
    lines += [
        "",
        "Calibration-size and all arm-wise errors are retained in the JSON artifact.",
        "Rows are not pooled across seeds.",
        "Claim boundary: targeted repair diagnostic, not evidence for the unmodified frozen model's causal control or planning ability.",
        "",
    ]
    (output / "contact_predictor_deep_summary.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
