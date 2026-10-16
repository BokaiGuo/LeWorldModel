"""Audit existing candidate packages for D1 teacher eligibility.

This is a read-only provenance audit.  It never repairs labels, creates a
teacher score, or promotes a package to D2.  A package is eligible only when
teacher identity, leakage, repeated-context stability structure, and runtime
references all pass independently.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected JSON object with records")
    return dict(payload.get("metadata", {})), list(payload["records"])


def audit_package(path: Path) -> dict[str, Any]:
    metadata, rows = _load(path)
    teacher_models = sorted({str(row.get("teacher_model_id", metadata.get("teacher_model_id", "missing"))) for row in rows})
    generators = sorted({str(row.get("candidate_generator", "missing")) for row in rows})
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    roots = {str(row.get("instance_id", "missing")) for row in rows}
    contexts = {
        str(row.get("teacher_set_id", row.get("candidate_bank_context_id", "missing")))
        for row in rows
    }
    exact_residuals = []
    numeric_teacher_cost = True
    for row in rows:
        try:
            exact_residuals.append(float(row["teacher_score"]) + float(row["real_task_cost"]))
        except (KeyError, TypeError, ValueError):
            numeric_teacher_cost = False
            break
    exact_leakage = bool(numeric_teacher_cost and exact_residuals and np.allclose(exact_residuals, 0.0, rtol=0.0, atol=1.0e-10))

    pair_contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing_runtime_refs = 0
    for row in rows:
        root = str(row.get("instance_id", "missing"))
        candidate = str(row.get("matched_action_hash", row.get("candidate_id", "missing")))
        context = str(row.get("candidate_bank_context_id", row.get("teacher_set_id", "missing")))
        pair_contexts[(root, candidate)].add(context)
        if row.get("split") in {"outcome", "environment_test"} and (
            row.get("predicted_latent_ref") is None or row.get("realized_latent_ref") is None
        ):
            missing_runtime_refs += 1
    repeated_pairs = sum(len(contexts_for_pair) >= 2 for contexts_for_pair in pair_contexts.values())
    teacher_is_external = len(teacher_models) == 1 and not teacher_models[0].startswith("offline_environment_cost") and teacher_models[0] != "missing"
    gates = {
        "single_external_teacher": teacher_is_external,
        "no_exact_negative_cost_leakage": not exact_leakage,
        "repeated_candidate_contexts": repeated_pairs > 0,
        "complete_outcome_runtime_refs": missing_runtime_refs == 0,
    }
    return {
        "path": str(path),
        "schema": "gdd_wm_teacher_candidate_package.v0.1",
        "summary": {
            "record_count": len(rows),
            "root_count": len(roots),
            "candidate_set_count": len(contexts),
            "teacher_models": teacher_models,
            "candidate_generators": generators,
            "split_counts": dict(sorted(split_counts.items())),
            "repeated_root_candidate_pairs": int(repeated_pairs),
            "missing_runtime_ref_count": int(missing_runtime_refs),
        },
        "teacher_execution_leakage": {
            "exact_negative_cost_match": exact_leakage,
            "numeric_teacher_cost_fields": numeric_teacher_cost,
        },
        "gates": gates,
        "eligible_for_d1_and_d2": bool(all(gates.values())),
        "claim_boundary": "Package eligibility audit only; no teacher correctness or planning claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("candidate_dataset.json"))
    if not paths:
        raise ValueError(f"no candidate_dataset.json found under {args.root}")
    packages = [audit_package(path) for path in paths]
    eligible = [package for package in packages if package["eligible_for_d1_and_d2"]]
    report = {
        "schema": "gdd_wm_teacher_candidate_audit.v0.1",
        "status": "completed_audit",
        "protocol": {
            "root": str(args.root),
            "package_count": len(packages),
            "outcome_used_for_selection": False,
            "teacher_labels_created": False,
        },
        "packages": packages,
        "summary": {
            "eligible_package_count": len(eligible),
            "eligible_paths": [package["path"] for package in eligible],
            "rejected_package_count": len(packages) - len(eligible),
        },
        "claim_boundary": "Read-only D1/D2 provenance audit; an eligible package still requires teacher correctness and independent held-out consequence validation.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "teacher_candidate_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
