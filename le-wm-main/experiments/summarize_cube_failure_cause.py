"""Join Cube ceiling and RGB panel artifacts into a predeclared cause audit.

This utility does not select a preferred arm.  It joins rows by the declared
root seed and reports which necessary condition failed first:

* ``support_failure``: even the expanded candidate bank cannot reach the goal;
* ``contact_failure``: the RGB arm fails to acquire contact;
* ``transfer_or_ranker_failure``: an expanded privileged bank is reachable
  and contact is observed, but the RGB arm does not achieve the goal;
* ``unresolved``: the available evidence is insufficient or contradictory.

The thresholds are protocol constants and are recorded in the output.  The
classification is a diagnosis, not evidence that a new controller works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTACT_RATE_THRESHOLD = 0.5
DEFAULT_SUCCESS_RATE_THRESHOLD = 0.5


def _root_map(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "seed" not in row:
            continue
        result[int(row["seed"])] = row
    return result


def classify_arm(
    *,
    ceiling: dict[str, Any],
    panel: dict[str, Any],
    contact_rate_threshold: float = DEFAULT_CONTACT_RATE_THRESHOLD,
    success_rate_threshold: float = DEFAULT_SUCCESS_RATE_THRESHOLD,
) -> dict[str, Any]:
    """Classify one RGB arm using only paired roots present in both artifacts."""
    ceiling_rows = _root_map(ceiling.get("records", []))
    panel_rows = _root_map(panel.get("test", []))
    joined = []
    for seed in sorted(set(ceiling_rows) & set(panel_rows)):
        c = ceiling_rows[seed]
        p = panel_rows[seed]
        expanded = c.get("expanded", {})
        factorized = c.get("factorized", expanded)
        multiphase = c.get("multiphase", factorized)
        row = {
            "seed": seed,
            "fixed_reachable": bool(c.get("fixed", {}).get("reachable", False)),
            "expanded_reachable": bool(expanded.get("reachable", False)),
            "factorized_reachable": bool(factorized.get("reachable", False)),
            "multiphase_reachable": bool(multiphase.get("reachable", False)),
            "contact_observed": bool(p.get("contact_observed", False)),
            "success": bool(p.get("success", False)),
        }
        # The first failed necessary condition is the most informative
        # per-root label.  Keep this separate from the aggregate status: a
        # panel can contain both planner-unreachable and transfer failures.
        # The strongest declared support ceiling is multiphase.  Legacy
        # artifacts fall back to factorized/expanded above, so this remains
        # backward compatible while allowing a genuine support repair to
        # change the downstream attribution.
        if not row["multiphase_reachable"]:
            row["failure_cause"] = "support_failure"
        elif not row["contact_observed"]:
            row["failure_cause"] = "contact_failure"
        elif not row["success"]:
            row["failure_cause"] = "transfer_or_ranker_failure"
        else:
            row["failure_cause"] = "success"
        joined.append(row)
    if not joined:
        return {
            "status": "unresolved",
            "reason": "no root seeds shared by ceiling and panel artifacts",
            "paired_roots": 0,
            "rows": [],
        }
    fixed_unreachable = [row for row in joined if not row["fixed_reachable"]]
    support_reachable = [row for row in joined if row["multiphase_reachable"]]
    no_contact = [row for row in joined if not row["contact_observed"]]
    no_success = [row for row in joined if not row["success"]]
    support_unreachable = [row for row in joined if not row["multiphase_reachable"]]
    failure_counts = {
        cause: sum(row["failure_cause"] == cause for row in joined)
        for cause in (
            "support_failure",
            "contact_failure",
            "transfer_or_ranker_failure",
            "success",
        )
    }
    dominant_failure = max(
        (cause for cause in failure_counts if cause != "success"),
        key=lambda cause: failure_counts[cause],
        default=None,
    )
    if len(support_unreachable) == len(joined):
        status = "support_failure"
        reason = "strongest declared candidate bank is unreachable on every paired root"
    elif len(no_contact) / len(joined) >= 1.0 - contact_rate_threshold:
        status = "contact_failure"
        reason = "RGB arm fails contact acquisition on the declared majority threshold"
    elif len(support_reachable) / len(joined) >= success_rate_threshold and len(no_success) > 0:
        status = "transfer_or_ranker_failure"
        reason = "expanded privileged support is reachable while RGB success remains incomplete"
    else:
        status = "unresolved"
        reason = "support, contact, and downstream outcomes do not isolate one cause"
    return {
        "status": status,
        "reason": reason,
        "paired_roots": len(joined),
        "fixed_reachability_rate": sum(row["fixed_reachable"] for row in joined) / len(joined),
        "expanded_reachability_rate": sum(row["expanded_reachable"] for row in joined) / len(joined),
        "factorized_reachability_rate": sum(row["factorized_reachable"] for row in joined) / len(joined),
        "reachability_recovered_by_factorized_count": sum(
            bool(row["factorized_reachable"] and not row["fixed_reachable"])
            for row in joined
        ),
        "multiphase_reachability_rate": sum(
            row["multiphase_reachable"] for row in joined
        ) / len(joined),
        "reachability_recovered_by_multiphase_count": sum(
            bool(row["multiphase_reachable"] and not row["fixed_reachable"])
            for row in joined
        ),
        "fixed_bank_only_support_gap_rate": sum(
            bool(row["expanded_reachable"] and not row["fixed_reachable"])
            for row in joined
        ) / len(joined),
        "contact_rate": sum(row["contact_observed"] for row in joined) / len(joined),
        "success_rate": sum(row["success"] for row in joined) / len(joined),
        "failure_counts": failure_counts,
        "dominant_failure": dominant_failure,
        "rows": joined,
    }


def build_report(
    *,
    ceiling: dict[str, Any],
    panel: dict[str, Any],
    contact_rate_threshold: float = DEFAULT_CONTACT_RATE_THRESHOLD,
    success_rate_threshold: float = DEFAULT_SUCCESS_RATE_THRESHOLD,
) -> dict[str, Any]:
    arms = panel.get("arms", {})
    # The per-arm raw test rows are stored in each artifact path by the panel.
    # The caller may provide them under ``arm_payloads`` for a lossless join.
    payloads = panel.get("arm_payloads", {})
    classifications = {
        arm: classify_arm(
            ceiling=ceiling,
            panel=payloads[arm],
            contact_rate_threshold=contact_rate_threshold,
            success_rate_threshold=success_rate_threshold,
        )
        for arm in arms
        if arm in payloads
    }
    return {
        "schema": "cube_failure_cause_audit.v1",
        "thresholds": {
            "contact_rate_threshold": contact_rate_threshold,
            "success_rate_threshold": success_rate_threshold,
        },
        "classifications": classifications,
        "claim_boundary": "predeclared failure attribution; not a controller or a positive planning result",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceiling", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ceiling = json.loads(args.ceiling.read_text(encoding="utf-8"))
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    # The panel summary intentionally does not duplicate raw arm rows.  Read
    # the paths it recorded and attach them only for this audit utility.
    arm_payloads = {}
    for arm, path in panel.get("artifacts", {}).items():
        arm_payloads[arm] = json.loads(Path(path).read_text(encoding="utf-8"))
    panel_for_audit = dict(panel)
    panel_for_audit["arm_payloads"] = arm_payloads
    report = build_report(ceiling=ceiling, panel=panel_for_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
