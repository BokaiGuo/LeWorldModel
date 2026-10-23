"""Aggregate fixed-protocol PointMaze v2 repairability contexts conservatively.

Candidate contexts may share roots, so this runner deliberately does *not*
pool them as independent samples or report a confidence interval.  It is a
transport diagnostic: did a pre-outcome router retain its conclusion as the
shared candidate-action context changed?
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "pointmaze_repairability_frontier_v2_context_aggregate.v1"
REPORT_SCHEMA = "pointmaze_execution_grounded_repairability_frontier.v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{path} is not a v2 repairability report")
    if value.get("control_contract", {}).get("outcome_execution_deferred_until_frozen") is not True:
        raise ValueError(f"{path} lacks deferred-outcome contract")
    if value.get("split_isolation", {}).get("root_disjoint") is not True:
        raise ValueError(f"{path} lacks four-way root isolation")
    if value.get("outcome", {}).get("outcome_cost_used_for_selection") is not False:
        raise ValueError(f"{path} used outcome costs for selection")
    return value


def aggregate(reports: list[Path]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("need at least two candidate-context reports")
    loaded = [(path.resolve(), _load(path.resolve())) for path in reports]
    first_roles = loaded[0][1]["split_isolation"]["roles"]
    role_sets = {role: sorted(values) for role, values in first_roles.items()}
    entries: list[dict[str, Any]] = []
    dataset_hashes: set[str] = set()
    for path, report in loaded:
        roles = {role: sorted(values) for role, values in report["split_isolation"]["roles"].items()}
        if roles != role_sets:
            raise ValueError("context reports must share the exact root-role assignment")
        provenance = report.get("provenance", {})
        dataset_hash = str(provenance.get("candidate_dataset_sha256", ""))
        if not dataset_hash:
            raise ValueError("context report lacks candidate dataset hash")
        if dataset_hash in dataset_hashes:
            raise ValueError("duplicate candidate context supplied")
        dataset_hashes.add(dataset_hash)
        outcome = report["outcome"]
        regret = outcome["mean_regret"]
        entries.append({
            "report": str(path),
            "report_sha256": _sha256(path),
            "candidate_dataset": provenance.get("candidate_dataset"),
            "candidate_dataset_sha256": dataset_hash,
            "selected_router_family": report["selector_configuration"]["selected_router_family"],
            "selected_router_gdd_auc": report["selector_configuration"]["selected_router_gdd_auc"],
            "crossfit_repair_prevalence": report["diagnosis"]["crossfit_label_prevalence"],
            "outcome_repair_prevalence": outcome["repair_outcome_prevalence"],
            "router_coverage": outcome["router_coverage"],
            "psd_regret": regret["psd"],
            "mlp_regret": regret["mlp"],
            "routed_regret": regret["routed"],
            "outcome_verdict": outcome["preselection_prediction_verdict"]["status"],
        })
    all_abstained = all(float(item["router_coverage"]) == 0.0 for item in entries)
    return {
        "schema": SCHEMA,
        "context_count": len(entries),
        "root_roles": role_sets,
        "contexts": entries,
        "transport_summary": {
            "all_contexts_abstained": all_abstained,
            "all_contexts_failed_preselection_verdict": all(
                item["outcome_verdict"] == "not_confirmed_in_this_pilot" for item in entries
            ),
            "mlp_beats_psd_context_count": sum(item["mlp_regret"] < item["psd_regret"] for item in entries),
            "routed_beats_psd_context_count": sum(item["routed_regret"] < item["psd_regret"] for item in entries),
        },
        "claim_boundary": (
            "A same-root, distinct-candidate-context transport diagnostic. Contexts share roots, "
            "so they are not independent replications and no pooled inferential claim is made."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["transport_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
