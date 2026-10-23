"""Audit a pre-frozen fixed-bank support-shift abstention rule for HTSC.

The preceding finite-sample HTSC audit found only 80% simultaneous
candidate-bank coverage on 100 external roots, below its declared 90% target.
This script asks a deliberately narrower question: can an *input-only*,
train/calibration-frozen latent-state OOD score identify an external region
where those coverage failures concentrate, while leaving a non-empty supported
subset?

It does not refit the effect adapter, alter its contrast bound, or inspect an
external realised successor before deciding support versus abstention.  The
existing external coverage labels are joined only after the rule is frozen.
Consequently, useful failure enrichment is a detection result, not a repaired
conformal certificate or a permission to execute actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.audit_htsc_finite_sample_calibration import (
    _canonical_action,
    _load_records,
)
from experiments.decision_benchmark.task_contrast import (
    _wilson_interval,
    calibrate_root_score_bound,
)


SCHEMA = "htsc.transport_support_shift_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_latent_rows(
    records: list[dict[str, Any]],
    *,
    allowed_splits: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Construct one frozen-latent state feature per root without outcomes.

    The declared action bank is world-fixed.  It is verified, but it cannot
    discriminate roots; the support score is therefore conditional on that
    exact bank and uses only the source latent state.  This avoids pretending
    that an action-only constant is a useful covariate-shift signal.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        split = str(row.get("split"))
        if split not in allowed_splits:
            continue
        groups[str(row["instance_id"])].append(row)
    if not groups:
        raise ValueError("no roots remain after split filtering")

    features: dict[str, np.ndarray] = {}
    bank_by_id: dict[str, str] = {}
    for root, rows in sorted(groups.items()):
        candidate_ids = [int(row["candidate_id"]) for row in rows]
        if len(candidate_ids) < 2 or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"root {root!r} needs two or more unique candidates")
        vectors = []
        for row in rows:
            candidate_id = str(int(row["candidate_id"]))
            action = _canonical_action(row)
            previous = bank_by_id.setdefault(candidate_id, action)
            if previous != action:
                raise ValueError("candidate bank changes across roots")
            vector = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("source latent must be non-empty and finite")
            vectors.append(vector)
        dimensions = {vector.size for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError(f"root {root!r} has inconsistent latent dimensions")
        features[root] = np.mean(np.stack(vectors, axis=0), axis=0)
    global_dimensions = {feature.size for feature in features.values()}
    if len(global_dimensions) != 1:
        raise ValueError("frozen latent dimensionality changes across roots")
    return features, bank_by_id


def fit_latent_mahalanobis_support(
    train_features: np.ndarray,
    *,
    relative_ridge: float = 1e-3,
) -> dict[str, Any]:
    """Fit a regularized train-only Gaussian support score.

    The ridge is relative to average coordinate variance and predeclared.  No
    calibration or outcome root participates in this density fit.
    """

    train = np.asarray(train_features, dtype=np.float64)
    if train.ndim != 2 or train.shape[0] < 2 or not np.isfinite(train).all():
        raise ValueError("train_features must be finite [roots, dimensions] with at least two roots")
    if not np.isfinite(relative_ridge) or relative_ridge <= 0.0:
        raise ValueError("relative_ridge must be positive and finite")
    mean = np.mean(train, axis=0)
    covariance = np.cov(train, rowvar=False, ddof=1)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    scale = float(np.trace(covariance) / covariance.shape[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("train covariance has no positive finite scale")
    ridge = float(relative_ridge * scale)
    precision = np.linalg.inv(covariance + ridge * np.eye(covariance.shape[0]))
    return {
        "mean": mean,
        "precision": precision,
        "relative_ridge": float(relative_ridge),
        "ridge": ridge,
        "latent_dimension": int(train.shape[1]),
        "fit_root_count": int(train.shape[0]),
    }


def mahalanobis_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    precision = np.asarray(model["precision"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != mean.size:
        raise ValueError("feature dimension does not match train-only support model")
    centered = values - mean
    scores = np.einsum("bi,ij,bj->b", centered, precision, centered)
    if not np.isfinite(scores).all() or np.any(scores < -1e-8):
        raise ValueError("Mahalanobis support scores must be finite and non-negative")
    return np.maximum(scores, 0.0)


def freeze_conformal_abstain_rule(
    calibration_scores: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Freeze an input-only abstain threshold before external outcomes exist."""

    calibration = calibrate_root_score_bound(calibration_scores, alpha=alpha)
    if calibration["bound"] is None:
        raise ValueError("insufficient calibration for a finite transport support threshold")
    rule = {
        "schema": "htsc.transport_support_rule.v1",
        "score": "train-only regularized frozen-latent Mahalanobis distance conditional on one fixed action bank",
        "decision": "support if score <= threshold; otherwise abstain",
        "threshold": float(calibration["bound"]),
        "alpha": float(alpha),
        "calibration": calibration,
        "pre_freeze_outcome_access": False,
        "claim_boundary": (
            "This is a marginal covariate-support rule under train/calibration exchangeability. "
            "It is not a conditional coverage certificate or an action release rule."
        ),
    }
    rule["rule_sha256"] = hashlib.sha256(
        json.dumps(rule, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rule


def _rate_summary(mask: np.ndarray, coverage_failure: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    failures = np.asarray(coverage_failure, dtype=bool)
    count = int(selected.sum())
    selected_failures = int(np.logical_and(selected, failures).sum())
    selected_covered = count - selected_failures
    return {
        "root_count": count,
        "coverage_failure_count": selected_failures,
        "coverage_failure_rate": None if count == 0 else float(selected_failures / count),
        "coverage_rate": None if count == 0 else float(selected_covered / count),
        "coverage_rate_wilson_95_interval": _wilson_interval(selected_covered, count),
    }


def stratify_frozen_rule(
    external_root_ids: list[str],
    external_scores: np.ndarray,
    frozen_rule: dict[str, Any],
    coverage_by_root: dict[str, bool],
) -> dict[str, Any]:
    """Join outcomes only after the support decisions are irrevocably frozen."""

    if len(external_root_ids) != len(external_scores):
        raise ValueError("external root IDs and scores must align")
    missing = sorted(set(external_root_ids) - set(coverage_by_root))
    if missing:
        raise ValueError(f"coverage labels missing for external roots: {missing[:3]}")
    scores = np.asarray(external_scores, dtype=np.float64)
    threshold = float(frozen_rule["threshold"])
    support = scores <= threshold
    failure = np.asarray([not bool(coverage_by_root[root]) for root in external_root_ids], dtype=bool)
    supported = _rate_summary(support, failure)
    abstained = _rate_summary(~support, failure)
    overall = _rate_summary(np.ones_like(support, dtype=bool), failure)
    total_failures = int(failure.sum())
    abstained_failures = int(np.logical_and(~support, failure).sum())
    support_failure_rate = supported["coverage_failure_rate"]
    abstain_failure_rate = abstained["coverage_failure_rate"]
    overall_failure_rate = overall["coverage_failure_rate"]
    enrichment = (
        None
        if abstain_failure_rate is None or overall_failure_rate in (None, 0.0)
        else float(abstain_failure_rate / overall_failure_rate)
    )
    return {
        "schema": "htsc.transport_support_stratification.v1",
        "outcome_join": "performed only after frozen support/abstain decisions",
        "threshold": threshold,
        "overall": overall,
        "supported": supported,
        "abstained": abstained,
        "detection_utility": {
            "failure_enrichment_in_abstained": enrichment,
            "failure_risk_difference_abstained_minus_supported": (
                None if abstain_failure_rate is None or support_failure_rate is None
                else float(abstain_failure_rate - support_failure_rate)
            ),
            "failure_capture_recall": (
                None if total_failures == 0 else float(abstained_failures / total_failures)
            ),
            "nonempty_supported_subset": bool(supported["root_count"] > 0),
            "enriches_failures": bool(
                abstain_failure_rate is not None
                and overall_failure_rate is not None
                and abstain_failure_rate > overall_failure_rate
            ),
            "interpretation": (
                "Detection utility is an empirical external stratification only. It does not create "
                "a conformal coverage guarantee for either stratum."
            ),
        },
        "certificate_validity": {
            "status": "not_established",
            "reason": (
                "The support threshold is calibrated on covariates, not on target-domain contrast errors. "
                "Conditioning on its selected subset changes the original fixed-bank calibration population. "
                "External coverage is therefore descriptive, not a renewed marginal or permit-conditional certificate."
            ),
            "original_external_target_coverage": 0.9,
            "supported_subset_meets_target_empirically": bool(
                supported["coverage_rate"] is not None and supported["coverage_rate"] >= 0.9
            ),
        },
    }


def run_audit(
    development_dataset: Path,
    external_dataset: Path,
    finite_sample_audit: Path,
    *,
    alpha: float,
    relative_ridge: float,
) -> dict[str, Any]:
    """Run the CPU-only post-hoc transport audit against a sealed result panel."""

    development = _load_records(development_dataset)
    external = _load_records(external_dataset)
    dev_features, dev_bank = _root_latent_rows(
        development, allowed_splits={"train", "calibration"},
    )
    external_features, external_bank = _root_latent_rows(
        external, allowed_splits={"external_outcome"},
    )
    if dev_bank != external_bank:
        raise ValueError("external candidate bank differs from the frozen development bank")
    split_by_root = {
        root: {str(row["split"]) for row in development if str(row["instance_id"]) == root}
        for root in dev_features
    }
    if any(len(splits) != 1 for splits in split_by_root.values()):
        raise ValueError("a development root appears in multiple splits")
    train_ids = sorted(root for root, splits in split_by_root.items() if splits == {"train"})
    calibration_ids = sorted(root for root, splits in split_by_root.items() if splits == {"calibration"})
    if not train_ids or not calibration_ids:
        raise ValueError("both train and calibration roots are required")
    model = fit_latent_mahalanobis_support(
        np.stack([dev_features[root] for root in train_ids]), relative_ridge=relative_ridge,
    )
    calibration_scores = mahalanobis_scores(
        np.stack([dev_features[root] for root in calibration_ids]), model,
    )
    frozen_rule = freeze_conformal_abstain_rule(calibration_scores, alpha=alpha)
    external_ids = sorted(external_features)
    external_scores = mahalanobis_scores(
        np.stack([external_features[root] for root in external_ids]), model,
    )

    # This is the first outcome read.  It is deliberately after score fitting,
    # calibration, threshold freeze, and support decisions above.
    audit_payload = json.loads(finite_sample_audit.read_text(encoding="utf-8"))
    roots = audit_payload["independent_external_root_coverage"]["roots"]
    coverage_by_root = {str(row["group_id"]): bool(row["covered"]) for row in roots}
    stratification = stratify_frozen_rule(
        external_ids, external_scores, frozen_rule, coverage_by_root,
    )
    return {
        "schema": SCHEMA,
        "runtime": {
            "device": "cpu",
            "cuda_used": False,
            "reason": (
                "No neural model is fit or refit: this audit uses a closed-form NumPy density score "
                "and an already-frozen finite-sample HTSC outcome artifact."
            ),
        },
        "protocol": {
            "score_fit_roots": "development train only",
            "threshold_roots": "development calibration only",
            "external_action_before_outcome_read": "frozen support or abstain decision per root",
            "external_outcome_read": "only the prior sealed coverage labels, after rule freeze",
            "support_feature": "frozen source latent state conditional on one verified world-fixed action bank",
            "action_bank_sha256": hashlib.sha256(
                json.dumps(dev_bank, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "action_bank_is_constant_across_roots": True,
            "outcome_labels_used_to_fit_score_or_threshold": False,
        },
        "provenance": {
            "development_dataset": str(development_dataset.resolve()),
            "development_dataset_sha256": _sha256(development_dataset),
            "external_dataset": str(external_dataset.resolve()),
            "external_dataset_sha256": _sha256(external_dataset),
            "sealed_finite_sample_audit": str(finite_sample_audit.resolve()),
            "sealed_finite_sample_audit_sha256": _sha256(finite_sample_audit),
        },
        "support_model": {
            key: value for key, value in model.items() if key not in {"mean", "precision"}
        },
        "frozen_rule": frozen_rule,
        "score_summary": {
            "train": mahalanobis_scores(np.stack([dev_features[root] for root in train_ids]), model).tolist(),
            "calibration": calibration_scores.tolist(),
            "external_root_ids": external_ids,
            "external": external_scores.tolist(),
        },
        "stratification": stratification,
        "claim_boundary": (
            "This is a constrained fixed-bank covariate-support diagnostic, not a generic HTSC threshold. "
            "It may show whether a pre-frozen latent OOD rule enriches external coverage failures while retaining support, "
            "but cannot itself validate a transport certificate, permit-conditional safety, action-conditional coverage, "
            "adaptive planning, RGB-only deployment, or closed-loop control benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--external-dataset", type=Path, required=True)
    parser.add_argument("--finite-sample-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--relative-ridge", type=float, default=1e-3)
    args = parser.parse_args()
    if not 0.0 < float(args.alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    result = run_audit(
        args.development_dataset,
        args.external_dataset,
        args.finite_sample_audit,
        alpha=float(args.alpha),
        relative_ridge=float(args.relative_ridge),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    utility = result["stratification"]["detection_utility"]
    print(json.dumps({
        "output": str(args.output),
        "threshold": result["frozen_rule"]["threshold"],
        "supported_roots": result["stratification"]["supported"]["root_count"],
        "abstained_roots": result["stratification"]["abstained"]["root_count"],
        "failure_enrichment_in_abstained": utility["failure_enrichment_in_abstained"],
        "enriches_failures": utility["enriches_failures"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
