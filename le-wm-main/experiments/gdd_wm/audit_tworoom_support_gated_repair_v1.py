"""Audit fresh root-disjoint support-gated TwoRoom repair panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_tworoom_action_effect_v4_panels import (
    EXPECTED_PANELS,
    _bootstrap,
    _ranking_delta,
    _load,
    _paired,
    audit as audit_base_panels,
)


SCHEMA = "tworoom_support_gated_repair_outcome_audit.v1"


def audit(report_paths: list[Path]) -> dict[str, Any]:
    base = audit_base_panels(report_paths)
    if len(report_paths) != EXPECTED_PANELS:
        raise ValueError("requires exactly the three predeclared panels")
    distances: dict[str, list[float]] = {"action_only": [], "effect_adapter": []}
    successes: dict[str, list[float]] = {"action_only": [], "effect_adapter": []}
    regrets: dict[str, list[float]] = {"action_only": [], "effect_adapter": []}
    releases: list[dict[str, Any]] = []
    for path in report_paths:
        report = _load(path)
        protocol = report["protocol"]
        receipt = report.get("support_gate")
        if protocol.get("support_gated_repair") is not True or not isinstance(receipt, dict):
            raise ValueError(f"{path}: support-gated protocol/receipt missing")
        if receipt.get("outcome_labels_read") is not False:
            raise ValueError(f"{path}: support gate read outcome labels")
        releases.append({
            "report": str(path.resolve()),
            "status": receipt.get("status"),
            "selected": receipt.get("selected"),
        })
        for index, control in enumerate(distances):
            roots, distance = _paired(report, control, "support_gated", "final_distance")
            _, success = _paired(report, "support_gated", control, "success")
            regret_roots, regret = _ranking_delta_for_gated(report, control)
            if roots != regret_roots:
                raise ValueError(f"{path}: ranking/closed-loop root order differs")
            distances[control].extend(distance.tolist())
            successes[control].extend(success.tolist())
            regrets[control].extend(regret.tolist())
    comparisons = {
        control: {
            "gated_closed_loop_distance_gain": _bootstrap(np.asarray(values), seed=202610271 + index),
            "gated_closed_loop_success_gain": _bootstrap(np.asarray(successes[control]), seed=202610281 + index),
            "gated_ranking_regret_gain": _bootstrap(np.asarray(regrets[control]), seed=202610291 + index),
        }
        for index, (control, values) in enumerate(distances.items())
    }
    return {
        "schema": SCHEMA,
        "status": "completed_root_disjoint_support_gated_audit",
        "base_panel_audit": base,
        "support_gate_releases": releases,
        "comparisons": comparisons,
        "claim_boundary": (
            "Root-disjoint TwoRoom support-gated outcome audit. It is not a conditional coverage guarantee, "
            "cross-environment result, or universal world-model claim."
        ),
    }


def _ranking_delta_for_gated(report: dict[str, Any], control: str) -> tuple[list[int], np.ndarray]:
    roots = [int(root) for root in report["protocol"]["root_ids"]["outcome"]]
    gated = report["ranking"]["support_gated"]["outcome"]["root_regret"]
    base = report["ranking"][control]["outcome"]["root_regret"]
    expected = {str(root) for root in roots}
    if set(gated) != expected or set(base) != expected:
        raise ValueError("ranking regret roots do not match declared outcome roots")
    return roots, np.asarray([float(base[str(root)]) - float(gated[str(root)]) for root in roots])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite support-gated audit: {args.output}")
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
