"""Summarize normalization robustness for the external conditional geometry audit."""

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
        raise ValueError("at least two normalization reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    if not all(_finite(report) for report in reports):
        raise ValueError("a normalization report contains non-finite values")
    reference = reports[0]
    identity = (reference["source"]["episodes"], reference["source"]["windows"])
    for report in reports[1:]:
        current = (report["source"]["episodes"], report["source"]["windows"])
        if current != identity:
            raise ValueError(f"inconsistent episode/window count: {identity} versus {current}")

    records: dict[str, Any] = {}
    for source in ("true", "predicted"):
        records[source] = {}
        for condition in ("magnitude", "direction"):
            heterogeneity = [
                _metric(report, source, condition, "conditional_minus_random/operator_heterogeneity")
                for report in reports
            ]
            commutator = [
                _metric(report, source, condition, "conditional_minus_random/commutator")
                for report in reports
            ]
            test_r2 = [
                _metric(report, source, condition, "conditional_minus_random/test_r2")
                for report in reports
            ]
            state_matched_heterogeneity = [
                _metric(report, source, condition, "conditional_minus_state_matched/operator_heterogeneity")
                for report in reports
            ]
            state_matched_test_r2 = [
                _metric(report, source, condition, "conditional_minus_state_matched/test_r2")
                for report in reports
            ]
            records[source][condition] = {
                "heterogeneity_delta": heterogeneity,
                "commutator_delta": commutator,
                "test_r2_delta": test_r2,
                "state_matched_heterogeneity_delta": state_matched_heterogeneity,
                "state_matched_test_r2_delta": state_matched_test_r2,
                "heterogeneity_mean_range": [
                    min(item["mean"] for item in heterogeneity),
                    max(item["mean"] for item in heterogeneity),
                ],
                "heterogeneity_all_ci_lower_positive": all(
                    item["ci95"][0] > 0 for item in heterogeneity
                ),
                "heterogeneity_sign_consistent": all(item["mean"] > 0 for item in heterogeneity),
                "state_matched_heterogeneity_mean_range": [
                    min(item["mean"] for item in state_matched_heterogeneity),
                    max(item["mean"] for item in state_matched_heterogeneity),
                ],
                "state_matched_heterogeneity_all_ci_lower_positive": all(
                    item["ci95"][0] > 0 for item in state_matched_heterogeneity
                ),
                "test_r2_mean_range": [
                    min(item["mean"] for item in test_r2),
                    max(item["mean"] for item in test_r2),
                ],
                "test_r2_all_ci_lower_positive": all(item["ci95"][0] > 0 for item in test_r2),
                "state_matched_test_r2_mean_range": [
                    min(item["mean"] for item in state_matched_test_r2),
                    max(item["mean"] for item in state_matched_test_r2),
                ],
                "state_matched_test_r2_all_ci_lower_positive": all(
                    item["ci95"][0] > 0 for item in state_matched_test_r2
                ),
            }

    normalization = [report["protocol"]["action_normalization"] for report in reports]
    report = {
        "artifact": "external_frozen_lewm_conditional_operator_geometry_normalization_robustness",
        "status": "cross_normalization_robustness_summary",
        "inputs": [str(path) for path in paths],
        "normalizations": normalization,
        "source": {"episodes": identity[0], "windows": identity[1]},
        "protocol_checks": {
            "same_episode_window_count": True,
            "covered_test_rows": [
                _metric(reports[0], "true", "magnitude", "conditional/covered_test_rows")["mean"],
                _metric(reports[0], "true", "magnitude", "random_partition/covered_test_rows")["mean"],
                _metric(reports[0], "true", "magnitude", "state_matched_partition/covered_test_rows")["mean"],
            ],
            "same_checkpoint_and_archive": len(
                {
                    (item["model"]["policy"], item["source"]["archive_sha256"])
                    for item in reports
                }
            )
            == 1,
        },
        "by_source": records,
        "claim_boundary": [
            "A positive heterogeneity delta with a positive bootstrap lower bound is evidence that action-aligned partitioning changes the fitted local transition map beyond a size-matched random partition.",
            "This robustness summary does not establish causal control, nonlinear Jacobian identification, online planning, task success, or improvement from training an alignment regularizer.",
        ],
    }
    if not _finite(report):
        raise ValueError("robustness report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# External conditional operator geometry: normalization robustness",
        "",
        f"Three frozen external runs use the same {identity[0]} episodes and {identity[1]} windows with normalization modes: {', '.join(normalization)}.",
        "",
        "| source | condition | ΔH vs random range | all ΔH CIs > 0 | ΔH vs state-matched range | all state-matched ΔH CIs > 0 | held-out R2 Δ range |",
        "| --- | --- | ---: | :---: | ---: | :---: | ---: |",
    ]
    for source in ("true", "predicted"):
        for condition in ("magnitude", "direction"):
            item = records[source][condition]
            h0, h1 = item["heterogeneity_mean_range"]
            s0, s1 = item["state_matched_heterogeneity_mean_range"]
            r0, r1 = item["test_r2_mean_range"]
            lines.append(
                f"| {source} | {condition} | [{h0:.6g}, {h1:.6g}] | "
                f"{item['heterogeneity_all_ci_lower_positive']} | [{s0:.6g}, {s1:.6g}] | "
                f"{item['state_matched_heterogeneity_all_ci_lower_positive']} | [{r0:.6g}, {r1:.6g}] |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The action-aligned local transition-map heterogeneity is the robust finding when every normalization has a positive episode-bootstrap lower bound, including the stronger state-matched null. Held-out R2 is reported separately: its sign is not used to promote the geometry diagnostic into a prediction or planning claim.",
        "",
        "This is frozen external evidence from a released checkpoint and validation set; it is not evidence for retraining gains, causal controllability, or online planning.",
    ])
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
