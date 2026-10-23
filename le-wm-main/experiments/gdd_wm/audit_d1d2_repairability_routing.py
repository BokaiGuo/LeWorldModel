"""Fail-closed D1/D2 routing audit for execution-supervised GDD controls.

This artifact is deliberately stricter than a generic ``D1 passed`` flag.  It
authorizes only a *small D2 diagnostic smoke*, never deployment, when all of
the following are bound to the same decision cell:

* a frozen, train-root-only execution-supervised teacher control;
* genuinely different repeated candidate-set contexts for the same actions;
* held roots with a declared fraction of action sequences absent from teacher
  training; and
* finite-intervention evidence with held-out probe verification.

It is intentionally impossible to reuse a permitted Reacher capacity audit to
open a PointMaze D2 run.  Missing provenance is a rejection rather than an
assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


SIGNATURE_FIELDS = (
    "environment",
    "world_model_sha256",
    "horizon",
    "candidate_family",
    "action_frame",
    "task_cost_version",
)
FORBIDDEN_TEACHER_INPUTS = {
    "candidate_id",
    "real_task_cost",
    "teacher_rank",
    "outcome_cost",
    "privileged_state",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_hash(row: dict[str, Any]) -> str:
    sequence = row.get("action_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(f"row {row.get('candidate_id')}: missing non-empty action_sequence")
    encoded = json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sign(value: float, margin: float = 1.0e-8) -> int:
    if abs(value) <= margin:
        return 0
    return 1 if value > 0.0 else -1


def _signature(metadata: dict[str, Any]) -> dict[str, Any] | None:
    value = metadata.get("routing_signature")
    if not isinstance(value, dict):
        return None
    if any(field not in value for field in SIGNATURE_FIELDS):
        return None
    return {field: value[field] for field in SIGNATURE_FIELDS}


def _teacher_control(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    roots = {
        split: {str(row["instance_id"]) for row in rows if str(row.get("split")) == split}
        for split in ("train", "calibration", "outcome")
    }
    teacher_models = {str(row.get("teacher_model_id", metadata.get("teacher_model_id", "missing"))) for row in rows}
    score_sources = {str(row.get("teacher_score_source", "missing")) for row in rows}
    input_fields = {str(value) for value in metadata.get("teacher_input_fields", [])}
    target_fields = {str(value) for value in metadata.get("teacher_training_target_fields", [])}
    fit_roots_value = metadata.get("teacher_fit_root_ids")
    fit_roots = {str(value) for value in fit_roots_value} if isinstance(fit_roots_value, list) else set()
    exact_negative_cost_match = bool(rows) and all(
        np.isclose(float(row["teacher_score"]) + float(row["real_task_cost"]), 0.0, rtol=0.0, atol=1.0e-10)
        for row in rows
    )
    forbidden_input_fields = sorted(input_fields & FORBIDDEN_TEACHER_INPUTS)
    forbidden_target_fields = sorted(target_fields & FORBIDDEN_TEACHER_INPUTS)
    direct_cost_source = any("real_task_cost" in source or "outcome_cost" in source for source in score_sources)
    checkpoint_sha256 = metadata.get("teacher_checkpoint_sha256")
    required_metadata = {
        "teacher_model_frozen_before_d1": metadata.get("teacher_model_frozen_before_d1") is True,
        "teacher_fit_split_train": metadata.get("teacher_fit_split") == "train",
        "teacher_fit_roots_exact_train_roots": bool(fit_roots) and fit_roots == roots["train"],
        "teacher_input_schema_declared": bool(input_fields),
        "teacher_training_targets_declared": bool(target_fields),
        "teacher_training_outcome_free": metadata.get("teacher_training_uses_execution_outcomes") is False,
        "teacher_checkpoint_pinned": isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64,
    }
    passed = bool(
        all(required_metadata.values())
        and len(teacher_models) == 1
        and "missing" not in teacher_models
        and not exact_negative_cost_match
        and not forbidden_input_fields
        and not forbidden_target_fields
        and not direct_cost_source
        and not (fit_roots & (roots["calibration"] | roots["outcome"]))
    )
    return {
        "passed": passed,
        "teacher_models": sorted(teacher_models),
        "teacher_score_sources": sorted(score_sources),
        "teacher_fit_roots": sorted(fit_roots),
        "split_root_counts": {split: len(values) for split, values in roots.items()},
        "exact_negative_cost_match": exact_negative_cost_match,
        "forbidden_input_fields": forbidden_input_fields,
        "forbidden_target_fields": forbidden_target_fields,
        "direct_cost_source": direct_cost_source,
        "metadata_checks": required_metadata,
    }


def _repeated_context_stability(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_contexts = int(metadata.get("candidate_context_count_required", 3))
    cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        root = str(row["instance_id"])
        context = str(row.get("candidate_bank_context_id", row["teacher_set_id"]))
        action = _action_hash(row)
        # A source export can retain two aliases for byte-identical action
        # sequences.  They are one decision, not a new pairwise comparison.
        # Canonicalize deterministically instead of turning a provenance audit
        # into an incidental candidate-ID collision.  Conflicting aliases are
        # still rejected: they would make the teacher ranking non-identifiable.
        prior = cells[root][context].get(action)
        if prior is not None:
            if not np.isclose(
                float(prior["teacher_score"]), float(row["teacher_score"]), rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"conflicting duplicate root/context/action triple: {root}/{context}/{action}")
            continue
        cells[root][context][action] = row

    roots_with_required_contexts = 0
    roots_with_genuine_contexts = 0
    stable_pairs = 0
    unstable_pairs = 0
    per_root: dict[str, Any] = {}
    for root, contexts in sorted(cells.items()):
        context_actions = [set(entries) for entries in contexts.values()]
        context_count = len(contexts)
        common_actions = set.intersection(*context_actions) if context_actions else set()
        unique_action_sets = {tuple(sorted(actions)) for actions in context_actions}
        has_required = context_count >= required_contexts
        genuinely_different = len(unique_action_sets) >= 2
        roots_with_required_contexts += int(has_required)
        roots_with_genuine_contexts += int(has_required and genuinely_different)
        root_stable = root_unstable = 0
        for left, right in combinations(sorted(common_actions), 2):
            signs = {
                _sign(float(context[left]["teacher_score"]) - float(context[right]["teacher_score"]))
                for context in contexts.values()
            }
            if 0 in signs:
                continue
            if len(signs) == 1:
                stable_pairs += 1
                root_stable += 1
            else:
                unstable_pairs += 1
                root_unstable += 1
        per_root[root] = {
            "context_count": context_count,
            "common_action_count": len(common_actions),
            "genuinely_different_contexts": genuinely_different,
            "stable_pair_count": root_stable,
            "unstable_pair_count": root_unstable,
        }
    passed = bool(
        cells
        and roots_with_genuine_contexts == len(cells)
        and stable_pairs > 0
        and unstable_pairs == 0
    )
    return {
        "passed": passed,
        "required_context_count": required_contexts,
        "root_count": len(cells),
        "roots_with_required_contexts": roots_with_required_contexts,
        "roots_with_genuinely_different_contexts": roots_with_genuine_contexts,
        "stable_pair_count": stable_pairs,
        "unstable_pair_count": unstable_pairs,
        "per_root": per_root,
    }


def _held_candidate_separation(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = {_action_hash(row) for row in rows if str(row.get("split")) == "train"}
    held = {_action_hash(row) for row in rows if str(row.get("split")) == "outcome"}
    novel = held - train
    minimum = float(metadata.get("held_candidate_min_novel_fraction", 0.25))
    fraction = None if not held else float(len(novel) / len(held))
    return {
        "passed": bool(held and fraction is not None and fraction >= minimum),
        "training_candidate_count": len(train),
        "held_candidate_count": len(held),
        "held_novel_candidate_count": len(novel),
        "held_novel_fraction": fraction,
        "required_minimum_novel_fraction": minimum,
    }


def _finite_intervention_repairability(
    signature: dict[str, Any] | None,
    dataset_sha256: str,
    evidence_path: Path | None,
) -> dict[str, Any]:
    if evidence_path is None:
        return {"passed": False, "reason": "missing_finite_intervention_evidence", "signature_matches": False}
    evidence = _load(evidence_path)
    evidence_signature = evidence.get("routing_signature")
    protocol = evidence.get("probe_protocol")
    signature_matches = isinstance(evidence_signature, dict) and signature is not None and all(
        evidence_signature.get(field) == signature.get(field) for field in SIGNATURE_FIELDS
    )
    dataset_matches = evidence.get("candidate_dataset_sha256") == dataset_sha256
    if not isinstance(protocol, dict):
        return {
            "passed": False,
            "reason": "missing_probe_protocol",
            "signature_matches": signature_matches,
            "candidate_dataset_matches": dataset_matches,
        }
    checks = {
        "schema": evidence.get("schema") == "gdd_wm_finite_intervention_repairability.v0.1",
        "signature_matches": signature_matches,
        "candidate_dataset_matches": dataset_matches,
        "minimum_probe_budget": int(protocol.get("probe_count", 0)) >= 6,
        "minimum_fit_probes": int(protocol.get("fit_probe_count", 0)) >= 4,
        "minimum_holdout_probes": int(protocol.get("holdout_probe_count", 0)) >= 2,
        "outcome_free_probe_selection": protocol.get("selection_uses_outcome") is False,
        "heldout_effect_validation": protocol.get("heldout_effect_validation_passed") is True,
        "local_support": protocol.get("local_support_passed") is True,
        "mode_stability": protocol.get("mode_stability_passed") is True,
        "robust_margin": protocol.get("margin_passed") is True,
    }
    return {
        "passed": bool(all(checks.values())),
        "evidence": str(evidence_path),
        "signature_matches": signature_matches,
        "candidate_dataset_matches": dataset_matches,
        "checks": checks,
        "probe_protocol": protocol,
    }


def audit(candidate_dataset: Path, repairability_evidence: Path | None = None) -> dict[str, Any]:
    """Audit one D1/D2 decision cell and return a non-deployment permission."""

    payload = _load(candidate_dataset)
    metadata = payload.get("metadata")
    rows = payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("candidate dataset requires non-empty metadata and records")
    required = {
        "instance_id", "split", "candidate_id", "teacher_set_id", "teacher_score",
        "real_task_cost", "action_sequence",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: expected object")
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"row {index}: missing {missing}")
        if str(row["split"]) not in {"train", "calibration", "outcome"}:
            raise ValueError(f"row {index}: split must be train/calibration/outcome")
        float(row["teacher_score"])
        float(row["real_task_cost"])

    signature = _signature(metadata)
    dataset_sha256 = _sha256(candidate_dataset)
    teacher = _teacher_control(metadata, rows)
    stability = _repeated_context_stability(metadata, rows)
    held = _held_candidate_separation(metadata, rows)
    intervention = _finite_intervention_repairability(signature, dataset_sha256, repairability_evidence)
    gates = {
        "routing_signature": signature is not None,
        "teacher_control": teacher["passed"],
        "repeated_context_stability": stability["passed"],
        "held_candidate_separation": held["passed"],
        "finite_intervention_repairability": intervention["passed"],
    }
    rejections = [name for name, passed in gates.items() if not passed]
    authorized = not rejections
    return {
        "schema": "gdd_wm_d1d2_repairability_routing.v0.1",
        "status": "permitted_small_d2_smoke" if authorized else "rejected_fail_closed",
        "permission": {
            "scope": "small D2 diagnostic smoke only; never a deployment or planning permission",
            "teacher_runtime_forbidden": True,
            "outcome_used_for_teacher_or_gate_selection": False,
            "candidate_dataset_sha256": dataset_sha256,
            "routing_signature": signature,
        },
        "teacher_control": teacher,
        "repeated_context_stability": stability,
        "held_candidate_separation": held,
        "finite_intervention_repairability": intervention,
        "gates": gates,
        "rejection_reasons": rejections,
        "gpu_smoke": {
            "authorized": authorized,
            "decision": "small_d2_smoke_permitted" if authorized else "not_started_fail_closed",
            "reason": (
                "all provenance, context, held-candidate, and finite-intervention gates passed"
                if authorized
                else "failed gates: " + ", ".join(rejections)
            ),
        },
        "claim_boundary": (
            "A permitted result only licenses a small, root-disjoint D2 diagnostic. "
            "It does not establish teacher correctness, geometry repair, prediction preservation, "
            "closed-loop benefit, or universal transfer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--repairability-evidence", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.candidate_dataset, args.repairability_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rejection_reasons": report["rejection_reasons"]}, indent=2))
    return 0 if report["status"] == "permitted_small_d2_smoke" else 2


if __name__ == "__main__":
    raise SystemExit(main())
