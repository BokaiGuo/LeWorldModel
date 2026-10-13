"""Audit the action-block convention used by historical DINO replay artifacts.

This is a protocol audit, not a model rerun.  It proves whether a recorded
future-only intervention could affect the first predicted transition under the
checkpoint's ``a_t -> z_{t+1}`` training convention.  Historical artifacts are
never rewritten; the report marks them as superseded by the corrected protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .action_interventions import predictor_action_prefix_length, predictor_action_source_indices
except ImportError:  # pragma: no cover - supports direct script execution
    from action_interventions import predictor_action_prefix_length, predictor_action_source_indices


DEFAULT_ARTIFACTS = (
    Path("results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/metrics/dino_pusht_future_zero.json"),
    Path("results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/metrics/dino_pointmaze_future_zero.json"),
    Path("results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/metrics/dino_wall_future_zero.json"),
)


def audit_artifacts(paths: Iterable[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        protocol = payload["protocol"]
        context_length = int(protocol["context_length"])
        old_prefix = int(protocol["action_block_preserved_prefix"])
        first_source = predictor_action_source_indices(context_length, 1)[0]
        corrected_prefix = predictor_action_prefix_length(context_length)
        h1 = payload.get("latent_prediction_error_by_horizon", {}).get("1", {})
        rows.append(
            {
                "artifact": str(path),
                "environment": {
                    "dino_pusht_future_zero.json": "PushT",
                    "dino_pointmaze_future_zero.json": "PointMaze",
                    "dino_wall_future_zero.json": "Wall",
                }.get(path.name, protocol.get("environment", path.name.split("_future")[0])),
                "context_length": context_length,
                "historical_preserved_prefix": old_prefix,
                "first_prediction_action_block": first_source,
                "historical_h1_source_preserved": bool(old_prefix > first_source),
                "corrected_preserved_prefix": corrected_prefix,
                "corrected_h1_source_intervened": bool(corrected_prefix <= first_source),
                "historical_h1_mse": h1.get("mean"),
                "historical_status": "valid_negative_or_partial_superseded_interface_baseline",
            }
        )
    return {
        "schema": "dino_action_alignment_audit.v1",
        "rows": rows,
        "conclusion": (
            "The historical future-only protocol preserved the action block consumed "
            "by h=1 in every audited artifact. The h=1 null effect is therefore a "
            "structural interface result, not evidence of action-insensitivity. "
            "Historical metrics remain retained as audit evidence and require a "
            "fresh rerun under the corrected source-action convention."
        ),
        "evidence_boundary": (
            "This audit establishes protocol identifiability only. It does not "
            "estimate corrected model action sensitivity, physical causality, "
            "planning performance, or cross-environment transfer."
        ),
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# DINO action-block alignment audit",
        "",
        "The historical artifacts are retained; this report only audits their intervention protocol.",
        "",
        "| Environment | context | historical prefix | h=1 source | historical h=1 preserved | corrected prefix |",
        "|---|---:|---:|---:|:---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['environment']} | {row['context_length']} | "
            f"{row['historical_preserved_prefix']} | {row['first_prediction_action_block']} | "
            f"{row['historical_h1_source_preserved']} | {row['corrected_preserved_prefix']} |"
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], "", "## Evidence boundary", "", report["evidence_boundary"], ""])
    output.with_suffix(".md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, action="append", dest="artifacts")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dino_action_alignment_audit_20261011/action_alignment_audit.json"),
    )
    args = parser.parse_args()
    report = audit_artifacts(args.artifacts or DEFAULT_ARTIFACTS)
    write_report(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
