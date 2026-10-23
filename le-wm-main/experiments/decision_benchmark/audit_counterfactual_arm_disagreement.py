"""Audit whether two counterfactual scoring arms ever make different decisions.

Mean regret differences are uninformative when two arms choose the same
candidate on most roots.  This post-execution audit separates the full panel
from the pre-outcome scorer-disagreement stratum without altering any selected
policy or accessing candidate features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "counterfactual_arm_disagreement_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aliasing_overlap(disagreement_roots: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    """Report whether empirical history aliasing actually explains arm changes.

    An alias witness and a counterfactual-arm disagreement are different
    events.  Their overlap is a post-execution mechanism check, never a
    routing signal or causal attribution.
    """

    audit_payload = payload.get("audit")
    if not isinstance(audit_payload, dict):
        raise ValueError("decision-aliasing artifact requires an audit object")
    top_pairs = audit_payload.get("top_pairs")
    evaluated = audit_payload.get("evaluated_query_count")
    if not isinstance(top_pairs, list) or not isinstance(evaluated, int) or len(top_pairs) != evaluated:
        raise ValueError("decision-aliasing artifact must retain every evaluated query pair for overlap analysis")
    alias_roots = {
        str(row.get("query_root_id", ""))
        for row in top_pairs
        if isinstance(row, dict) and row.get("near_decision_alias") is True and str(row.get("query_root_id", ""))
    }
    changed = set(disagreement_roots)
    overlap = sorted(alias_roots & changed)
    return {
        "alias_root_count": len(alias_roots), "arm_disagreement_root_count": len(changed),
        "overlap_root_count": len(overlap), "overlap_roots": overlap,
        "fraction_of_alias_roots_with_arm_disagreement": None if not alias_roots else float(len(overlap) / len(alias_roots)),
        "fraction_of_arm_disagreements_with_alias_witness": None if not changed else float(len(overlap) / len(changed)),
        "claim_boundary": "post-execution set overlap only; it cannot establish that history aliasing caused an arm decision or regret difference.",
    }


def _choice_diversity(extracted: list[tuple[str, str, str, float]], *, arm_index: int) -> dict[str, Any]:
    """Describe candidate-choice concentration after the panel is opened.

    A repair arm cannot lower regret by changing action ranking when its
    predicted winner is effectively constant across roots.  This is a
    descriptive post-execution explanation of a null arm difference, not a
    pre-outcome gate or a justification to modify a frozen candidate bank.
    """

    if arm_index not in (1, 2):
        raise ValueError("arm_index must select the left or right candidate identity")
    choices = [row[arm_index] for row in extracted]
    counts = {candidate: choices.count(candidate) for candidate in sorted(set(choices))}
    frequencies = np.asarray(list(counts.values()), dtype=np.float64) / len(choices)
    entropy = float(-(frequencies * np.log(frequencies)).sum()) if len(frequencies) else 0.0
    normalized_entropy = float(entropy / np.log(len(frequencies))) if len(frequencies) > 1 else 0.0
    return {
        "root_count": len(choices),
        "distinct_winner_count": len(counts),
        "winner_counts": counts,
        "dominant_winner_fraction": float(frequencies.max()) if len(frequencies) else None,
        "winner_entropy_normalized": normalized_entropy,
        "claim_boundary": "post-execution winner-concentration diagnostic only; it does not establish why a scorer made a choice or authorize a new candidate bank.",
    }


def audit(report: dict[str, Any], *, left_arm: str, right_arm: str,
          decision_aliasing: dict[str, Any] | None = None) -> dict[str, Any]:
    outcome = report.get("outcome")
    rows = outcome.get("per_root") if isinstance(outcome, dict) else None
    if not isinstance(rows, dict) or not rows:
        raise ValueError("report requires non-empty outcome.per_root")
    extracted: list[tuple[str, str, str, float]] = []
    for root, value in sorted(rows.items()):
        if not isinstance(value, dict):
            raise ValueError("each per-root result must be an object")
        left_id, right_id = value.get(f"{left_arm}_candidate_id"), value.get(f"{right_arm}_candidate_id")
        left_regret, right_regret = value.get(f"{left_arm}_regret"), value.get(f"{right_arm}_regret")
        try:
            regrets = np.asarray([left_regret, right_regret], dtype=np.float64)
        except (TypeError, ValueError):
            regrets = np.asarray([np.nan, np.nan], dtype=np.float64)
        if left_id is None or right_id is None or not np.isfinite(regrets).all():
            raise ValueError(f"{root}: missing finite arm identity or regret")
        extracted.append((str(root), str(left_id), str(right_id), float(regrets[1] - regrets[0])))
    disagreement = [value for value in extracted if value[1] != value[2]]
    def summarize(values: list[tuple[str, str, str, float]]) -> dict[str, Any]:
        delta = np.asarray([value[3] for value in values], dtype=np.float64)
        return {
            "root_count": len(values),
            "mean_right_minus_left_regret": float(delta.mean()) if len(delta) else None,
            "right_better_count": int((delta < 0.0).sum()),
            "left_better_count": int((delta > 0.0).sum()),
            "tie_count": int(np.isclose(delta, 0.0, rtol=0.0, atol=1.0e-12).sum()),
        }
    result = {
        "schema": SCHEMA, "left_arm": left_arm, "right_arm": right_arm,
        "full_panel": summarize(extracted), "disagreement_stratum": summarize(disagreement),
        "same_candidate_count": len(extracted) - len(disagreement),
        "candidate_disagreement_count": len(disagreement),
        "candidate_disagreement_coverage": float(len(disagreement) / len(extracted)),
        "disagreement_roots": [value[0] for value in disagreement],
        "left_arm_choice_diversity": _choice_diversity(extracted, arm_index=1),
        "right_arm_choice_diversity": _choice_diversity(extracted, arm_index=2),
        "claim_boundary": "post-execution decision-mechanism decomposition; not a policy selector, conditional guarantee, or causal attribution of an arm difference.",
    }
    if decision_aliasing is not None:
        result["history_aliasing_overlap"] = _aliasing_overlap(result["disagreement_roots"], decision_aliasing)
    return result


def run(*, report_path: Path, output: Path, left_arm: str, right_arm: str,
        decision_aliasing_audit: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not report_path.is_file():
        raise FileNotFoundError(f"report does not exist: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report must be a JSON object")
    alias_payload = None
    alias_provenance = None
    if decision_aliasing_audit is not None:
        if not decision_aliasing_audit.is_file():
            raise FileNotFoundError(f"decision-aliasing audit does not exist: {decision_aliasing_audit}")
        alias_payload = json.loads(decision_aliasing_audit.read_text(encoding="utf-8"))
        if not isinstance(alias_payload, dict):
            raise ValueError("decision-aliasing artifact must be a JSON object")
        alias_provenance = {"path": str(decision_aliasing_audit.resolve()), "sha256": _sha256(decision_aliasing_audit)}
    result = {
        **audit(payload, left_arm=left_arm, right_arm=right_arm, decision_aliasing=alias_payload),
        "status": "completed_post_execution_arm_disagreement_audit",
        "provenance": {"report": str(report_path.resolve()), "report_sha256": _sha256(report_path)},
    }
    if alias_provenance is not None:
        result["provenance"]["decision_aliasing_audit"] = alias_provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--left-arm", required=True)
    parser.add_argument("--right-arm", required=True)
    parser.add_argument("--decision-aliasing-audit", type=Path, default=None,
                        help="completed post-execution aliasing artifact; optional overlap diagnostic only")
    args = parser.parse_args()
    result = run(report_path=args.report, output=args.output, left_arm=args.left_arm, right_arm=args.right_arm,
                 decision_aliasing_audit=args.decision_aliasing_audit)
    print(json.dumps({"status": result["status"], "candidate_disagreement_count": result["candidate_disagreement_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
