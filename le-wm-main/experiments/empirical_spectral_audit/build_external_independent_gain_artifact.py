"""Build a strict external-consequence coverage artifact.

This adapter only joins already-produced, root-disjoint evaluator artifacts.
It never invents a missing metric and never converts an inapplicable metric to
zero.  The output is deliberately allowed to remain blocked; the readiness
audit accepts it only when every required metric is available with a held-out
estimate and paired interval under the required provenance protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = (
    "action_ranking_inversion",
    "multistep_task_error",
    "contact_retention",
    "angle_closure",
    "planning_regret",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _metric_from_source(payload: dict[str, Any], field: str) -> dict[str, Any] | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(field), dict):
        return dict(metrics[field])
    return None


def _coverage_from_reacher(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read(path)
    if payload.get("schema") != "reacher_shared_candidate_ranking.v1":
        raise ValueError(f"{path}: unexpected schema")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: metrics must be an object")
    extracted = {field: dict(metrics[field]) for field in metrics if isinstance(metrics.get(field), dict)}
    protocol = payload.get("protocol", {})
    provenance = {
        "environment": "Reacher",
        "source_artifact": str(path),
        "root_disjoint": bool(protocol.get("root_disjoint")),
        "runtime_state_access": bool(protocol.get("runtime_state_access", True)),
        "test_planning_used_for_selection": bool(protocol.get("test_planning_used_for_selection", True)),
        "checkpoint_frozen": bool(protocol.get("checkpoint_frozen", False)),
    }
    return extracted, provenance


def _blocked(reason: str, source: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "blocked", "reason": reason}
    if source is not None:
        result["source_artifact"] = source
    return result


def build(reacher: Path, output: Path) -> dict[str, Any]:
    reacher_metrics, reacher_provenance = _coverage_from_reacher(reacher)
    metrics: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        candidate = reacher_metrics.get(field)
        if candidate is not None and candidate.get("status") == "available":
            metrics[field] = candidate
        elif candidate is not None and candidate.get("status") == "not_applicable":
            metrics[field] = _blocked("metric_not_applicable_in_reacher", str(reacher))
        else:
            metrics[field] = _blocked("metric_missing_from_source", str(reacher))

    # These flags are intentionally explicit.  The Reacher ranking audit uses
    # held-out replay labels but is not yet a complete five-metric, cross-task
    # external gain artifact; this output must therefore remain fail-closed.
    gate = {
        "schema": "external_independent_gain.v1",
        "protocol": {
            "privileged_state_used": False,
            "test_planning_used_for_selection": False,
            "train_calibration_test_root_disjoint": bool(reacher_provenance["root_disjoint"]),
            "independent_heldout_consequence": False,
            "metric_selection_frozen": True,
            "runtime_input": "rendered RGB current/goal plus action/history; evaluator labels offline",
        },
        "metrics": metrics,
        "sources": [reacher_provenance],
        "coverage": {
            "available_fields": [field for field, value in metrics.items() if value.get("status") == "available"],
            "blocked_fields": [field for field, value in metrics.items() if value.get("status") != "available"],
            "status": "blocked_incomplete_cross_task_metric_panel",
        },
        "claim_boundary": (
            "Coverage artifact only. Reacher supplies non-contact ranking and angle metrics; "
            "missing contact and independent-gain requirements are not imputed."
        ),
    }
    report = {
        "schema": "external_independent_gain_bundle.v1",
        "external_independent_gain": gate,
        "coverage": gate["coverage"],
        "sources": gate["sources"],
        "claim_boundary": gate["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.reacher, args.output)
    print(json.dumps(report["coverage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
