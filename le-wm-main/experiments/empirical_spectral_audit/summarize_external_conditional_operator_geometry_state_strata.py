"""Summarize state-stratification sensitivity for the frozen geometry audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _metric(report: dict[str, Any], source: str, condition: str, key: str) -> dict[str, Any]:
    return report["crossfit"]["summary"]["by_source"][source][condition][key]


def summarize(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two state-strata reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    if not all(_finite(report) for report in reports):
        raise ValueError("a state-strata report contains non-finite values")
    strata = [int(report["protocol"]["state_strata"]) for report in reports]
    if len(set(strata)) != len(strata):
        raise ValueError("state-strata inputs must be distinct")
    if any(report["protocol"]["action_normalization"] != "public" for report in reports):
        raise ValueError("state-strata sensitivity is defined for public normalization")
    identity = (reports[0]["source"]["episodes"], reports[0]["source"]["windows"])
    for report in reports[1:]:
        current = (report["source"]["episodes"], report["source"]["windows"])
        if current != identity:
            raise ValueError(f"inconsistent episode/window count: {identity} versus {current}")

    records: dict[str, Any] = {}
    for source in ("true", "predicted"):
        records[source] = {}
        for condition in ("magnitude", "direction"):
            heterogeneity = {
                str(state_strata): _metric(
                    report,
                    source,
                    condition,
                    "conditional_minus_state_matched/operator_heterogeneity",
                )
                for state_strata, report in zip(strata, reports)
            }
            test_r2 = {
                str(state_strata): _metric(
                    report, source, condition, "conditional_minus_state_matched/test_r2"
                )
                for state_strata, report in zip(strata, reports)
            }
            coverage = {
                str(state_strata): _metric(
                    report, source, condition, "conditional/covered_test_rows"
                )["mean"]
                for state_strata, report in zip(strata, reports)
            }
            records[source][condition] = {
                "heterogeneity_delta": heterogeneity,
                "test_r2_delta": test_r2,
                "coverage": coverage,
                "heterogeneity_all_ci_lower_positive": all(
                    item["ci95"][0] > 0 for item in heterogeneity.values()
                ),
                "bootstrap_repeats": sorted(
                    {int(item["bootstrap_repeats"]) for item in heterogeneity.values()}
                ),
            }

    report = {
        "artifact": "external_frozen_lewm_conditional_operator_geometry_state_strata_sensitivity",
        "status": "state_stratification_sensitivity_summary",
        "inputs": [str(path) for path in paths],
        "state_strata": strata,
        "action_normalization": "public",
        "source": {"episodes": identity[0], "windows": identity[1]},
        "by_source": records,
        "claim_boundary": [
            "Positive intervals across state-strata counts show robustness of the descriptive heterogeneity contrast to this coarse-null granularity.",
            "This sensitivity does not establish causal action effects, physical dynamics, prediction gains, online planning, or trained-model improvement.",
        ],
    }
    if not _finite(report):
        raise ValueError("state-strata summary contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# External conditional operator geometry: state-strata sensitivity",
        "",
        f"Public-normalization reports use the same {identity[0]} episodes and {identity[1]} windows; each cell uses 2,000 episode-cluster bootstrap replicates.",
        "",
        "| source | condition | " + " | ".join(f"strata={item}" for item in strata) + " | all lower bounds > 0 |",
        "| --- | --- | " + " | ".join([":---:"] * len(strata)) + " | :---: |",
    ]
    for source in ("true", "predicted"):
        for condition in ("magnitude", "direction"):
            item = records[source][condition]
            cells = []
            for state_strata in strata:
                metric = item["heterogeneity_delta"][str(state_strata)]
                cells.append(
                    f"{metric['mean']:.5f} [{metric['ci95'][0]:.5f}, {metric['ci95'][1]:.5f}]"
                )
            lines.append(
                f"| {source} | {condition} | "
                + " | ".join(cells)
                + f" | {item['heterogeneity_all_ci_lower_positive']} |"
            )
    lines.extend(
        [
            "",
            "The result is interpreted as sensitivity of a frozen, descriptive local-map contrast to the number of coarse state strata. Held-out R2 is retained in the JSON artifact and is not promoted to a predictive claim.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.inputs, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
