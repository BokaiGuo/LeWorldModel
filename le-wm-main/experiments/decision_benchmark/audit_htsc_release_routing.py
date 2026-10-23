"""Calibration-only HTSC release-failure routing and recovery audit.

The audit is intentionally a diagnostic, not a threshold search.  It uses the
same frozen PointMaze candidate table as the matched-transfer evaluator and
first calibrates the contrast and model-risk gates on calibration roots only.
On test roots it preserves the evaluator's exact HTSC permit mask and matched
fallback action.  Non-permitted roots are *routed* to a proposed recovery
operation, never silently released:

* support failure -> abstain and collect new support;
* uncertainty failure -> re-observe/recalibrate uncertainty;
* certificate-margin failure -> short observed-effect probe;
* no gate failure -> permit.

True task costs on test roots are read only after this routing for a post-hoc
utility/ranking diagnostic.  They do not tune thresholds, alter the fallback
mask, or select an action.  This remains an offline PointMaze smoke because
the source fallback/support proxy is evaluator-state based.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.decision_benchmark.run_htsc_matched_transfer import (
    _pointmaze_records,
    evaluate_records,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "htsc.calibration_only_release_routing.v1"


def recovery_route(gate_failure_vector: dict[str, bool]) -> str:
    """Return a predeclared non-release recovery route from model-only gates.

    Priority is conservative: unavailable support invalidates uncertainty and
    contrast interpretation; uncertainty requires a new observation before a
    contrast probe; only an isolated certificate-margin failure receives the local
    observed-effect probe.  This function must not receive outcomes.
    """

    if bool(gate_failure_vector["support_failure"]):
        return "abstain_collect_new_support"
    if bool(gate_failure_vector["uncertainty_failure"]):
        return "reobserve_uncertainty"
    if bool(gate_failure_vector["certificate_margin_failure"]):
        return "short_observed_effect_probe"
    return "permit_frozen_htsc_choice"


def audit_report(report: dict[str, Any]) -> dict[str, Any]:
    """Annotate a frozen HTSC report without changing its decisions or mask."""

    roots = list(report["test"]["roots"])
    annotated: list[dict[str, Any]] = []
    for row in roots:
        vector = {key: bool(row["gate_failure_vector"][key]) for key in (
            "support_failure", "uncertainty_failure", "certificate_margin_failure",
        )}
        route = recovery_route(vector)
        # The action field repeats the evaluator decision precisely.  The
        # recovery route is diagnostic/future-work only; it cannot manufacture
        # a release or create an unmatched comparator.
        annotated.append({
            "root_id": str(row["root_id"]),
            "gate_failure_vector": vector,
            "route": route,
            "frozen_permit": bool(row["status"] == "permit"),
            "htsc_candidate_id": int(row["htsc_candidate_id"]),
            "matched_fallback_candidate_id": int(row["matched_fallback_candidate_id"]),
            "matched_fallback_mask_preserved": bool(
                int(row["htsc_candidate_id"]) != int(row["matched_fallback_candidate_id"])
            ) == bool(row["status"] == "permit"),
            "posthoc_utility_ranking_failure": bool(row["posthoc_utility_ranking_failure"]),
            "posthoc_coverage_failure": bool(row["posthoc_coverage_failure"]),
            "posthoc_wm_oracle_disagreement": bool(
                int(row["htsc_candidate_id"] if row["status"] == "permit" else row["matched_fallback_candidate_id"])
                != int(row["oracle_candidate_id"])
            ),
        })
    route_counts = Counter(row["route"] for row in annotated)
    failure_counts = {
        key: int(sum(bool(row["gate_failure_vector"][key]) for row in annotated))
        for key in ("support_failure", "uncertainty_failure", "certificate_margin_failure")
    }
    return {
        "schema": SCHEMA,
        "protocol": {
            "threshold_source": "contrast/risk thresholds are exactly the calibration-only thresholds of the embedded HTSC report",
            "test_tuned_threshold": False,
            "root_split": "train/calibration/test roots are inherited from, and validated by, the embedded fixed-bank evaluator",
            "matched_fallback_mask": "identical to HTSC permit mask; routes never alter selected actions",
            "recovery_semantics": "non-permit routes are diagnostic recommendations, not an action release",
        },
        "diagnosis": {
            "failure_counts": failure_counts,
            "route_counts": dict(sorted(route_counts.items())),
            "posthoc_utility_ranking_failure_count": int(sum(row["posthoc_utility_ranking_failure"] for row in annotated)),
            "posthoc_coverage_failure_count": int(sum(row["posthoc_coverage_failure"] for row in annotated)),
            "interpretation": (
                "Certificate-margin failure means the calibrated simultaneous contrast envelope exceeds the predicted winner margin; "
                "uncertainty failure means the calibration-only model-risk quantile failed; post-hoc coverage failure means the "
                "realized root contrast exceeded that envelope; utility/ranking failure is an outcome-only WM-winner/oracle "
                "disagreement. Neither outcome-only field can be repaired by changing a test threshold."
            ),
        },
        "root_routes": annotated,
        "embedded_htsc_report": report,
        "claim_boundary": (
            "This calibration-only routing audit does not establish permit-conditional safety, physical-cost calibration, "
            "online replanning, or RGB-only deployment. It identifies why a frozen fixed-bank HTSC evaluator abstains and "
            "keeps all non-permitted choices on the original matched fallback."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--risk-quantile", type=float, default=0.9)
    parser.add_argument("--max-test-roots", type=int, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026102201)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = require_cuda_allocation(args.device)
    records, provenance = _pointmaze_records(
        args.dataset, args.rgb_report, args.fit_export, max_test_roots=args.max_test_roots,
    )
    frozen_report = evaluate_records(
        records, alpha=float(args.alpha), risk_quantile=float(args.risk_quantile),
        bootstrap_draws=int(args.bootstrap_draws), seed=int(args.seed),
    )
    result = audit_report(frozen_report)
    result["runtime"] = runtime
    result["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "released_roots": result["embedded_htsc_report"]["test"]["released_root_count"],
        "failure_counts": result["diagnosis"]["failure_counts"],
        "route_counts": result["diagnosis"]["route_counts"],
        "posthoc_utility_ranking_failure_count": result["diagnosis"]["posthoc_utility_ranking_failure_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
