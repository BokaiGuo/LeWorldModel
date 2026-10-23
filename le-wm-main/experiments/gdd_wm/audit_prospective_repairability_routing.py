"""Evaluate whether pre-outcome evidence can route a visual-WM repair family.

This pilot turns a *post-execution* contrast audit into a lawful prospective
question.  The contrast labels are available only on train roots when fitting
the router.  A root-disjoint calibration split chooses and certifies a minimum
confidence threshold.  Outcome-root mechanism labels are read only after that
policy is frozen.

The routed labels are deliberately coarse and tied to the declared function
classes:

* ``metric_repair``: endpoint-readout or task-score geometry mismatch;
* ``dynamics_repair``: predicted-effect collapse or latent dynamics mismatch;
* ``abstain``: no dominant repairable mechanism under the audit taxonomy.

This is not a planner or a repair result.  It establishes only whether
decision-time features can prospectively identify a repair *family* better
than fixed all-metric/all-dynamics policies.  A later, separately frozen
intervention must validate that the selected family actually reduces regret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_prospective_repairability_routing.v0.1"
CLASSES = ("metric_repair", "dynamics_repair", "abstain")
FAMILY_REQUIRED_MODES = {
    "metric_repair": frozenset({"endpoint_readout_geometry_mismatch", "task_readout_mismatch"}),
    "dynamics_repair": frozenset({"predicted_action_effect_collapse", "predicted_dynamics_mismatch"}),
}
_EPS = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _fold(root: str) -> int:
    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 2


def _pairwise_upper(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("pairwise features require finite [candidate, dimension] values")
    distance = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    return distance[np.triu_indices(len(values), k=1)]


def _spread(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((values - values.mean(axis=0, keepdims=True)) ** 2, axis=1))))


def _root_features(rows: list[dict[str, Any]]) -> np.ndarray:
    """Build pre-outcome features without terminal labels or task costs."""

    ordered = sorted(rows, key=lambda row: str(row["candidate_id"]))
    if len(ordered) != 3:
        raise ValueError("pilot requires exactly three candidates per root/context decision group")
    predicted_xy = np.asarray([row["teacher_predicted_terminal_xy"] for row in ordered], dtype=np.float64)
    action = np.asarray([row["action_sequence"] for row in ordered], dtype=np.float64).reshape(len(ordered), -1)
    scores = np.asarray([float(row["teacher_score"]) for row in ordered], dtype=np.float64)
    source = np.asarray(ordered[0]["source_latent_model"], dtype=np.float64).reshape(-1)
    history = np.asarray(ordered[0]["source_history_model"], dtype=np.float64).reshape(-1)
    goal = np.asarray(ordered[0]["goal_raw"], dtype=np.float64).reshape(-1)
    if predicted_xy.shape != (3, 2) or action.shape != (3, 40) or source.size == 0 or history.size < source.size:
        raise ValueError("unexpected PointMaze candidate feature dimensions")
    if history.size % source.size or goal.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("visual history/action/goal contract is invalid")
    endpoint_pair = np.sort(_pairwise_upper(predicted_xy))
    action_pair = np.sort(_pairwise_upper(action))
    score_sorted = np.sort(scores)
    first, last = history[:source.size], history[-source.size:]
    return np.asarray([
        *endpoint_pair.tolist(), _spread(predicted_xy),
        *action_pair.tolist(), _spread(action),
        *score_sorted.tolist(), float(score_sorted[1] - score_sorted[0]),
        float(np.std(scores)), float(np.linalg.norm(last - first)),
        float(np.linalg.norm(last)), float(np.linalg.norm(source)), float(np.linalg.norm(goal)),
    ], dtype=np.float64)


def _map_label(mode: str) -> str:
    if mode in {"endpoint_readout_geometry_mismatch", "task_readout_mismatch"}:
        return "metric_repair"
    if mode in {"predicted_action_effect_collapse", "predicted_dynamics_mismatch"}:
        return "dynamics_repair"
    if mode == "no_dominant_failure_identified":
        return "abstain"
    raise ValueError(f"unknown contrast-audit failure mode: {mode}")


def _fit_centroid_router(features: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a deterministic, balanced nearest-centroid router on train roots."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("router train features must be finite and aligned")
    missing = sorted(set(CLASSES).difference(labels))
    if missing:
        raise ValueError(f"train roots lack repair classes {missing}; fail closed rather than collapsing classes")
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (values - mean) / scale
    centroids = np.stack([standardized[np.asarray([label == name for label in labels])].mean(axis=0) for name in CLASSES])
    return mean, scale, centroids


def _predict_centroid(features: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[list[str], np.ndarray]:
    mean, scale, centroids = fit
    values = np.asarray(features, dtype=np.float64)
    standardized = (values - mean) / scale
    squared = np.mean((standardized[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    logits = -squared
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    indices = np.argmax(probability, axis=1)
    return [CLASSES[int(index)] for index in indices], probability


def _policy(
    true: list[str], predicted: list[str], confidence: np.ndarray, threshold: float,
    *, allowed_families: set[str] | None = None, support: np.ndarray | None = None,
) -> dict[str, Any]:
    allowed = set(FAMILY_REQUIRED_MODES) if allowed_families is None else set(allowed_families)
    support_mask = np.ones(len(predicted), dtype=bool) if support is None else np.asarray(support, dtype=bool)
    if support_mask.shape != (len(predicted),):
        raise ValueError("support mask must align with routing rows")
    route = [
        guess if in_support and guess in allowed and float(conf) >= threshold else "abstain"
        for guess, conf, in_support in zip(predicted, confidence, support_mask)
    ]
    active = np.asarray([value != "abstain" for value in route], dtype=bool)
    correct = np.asarray([left == right for left, right in zip(true, route)], dtype=bool)
    utility = np.asarray([
        0.0 if selected == "abstain" else (1.0 if selected == observed else -1.0)
        for observed, selected in zip(true, route)
    ], dtype=np.float64)
    return {
        "route": route,
        "accuracy": float(correct.mean()),
        "active_coverage": float(active.mean()),
        "active_precision": None if not active.any() else float(correct[active].mean()),
        "support_coverage": float(support_mask.mean()),
        "mean_repair_utility": float(utility.mean()),
        "utility": utility,
    }


def _root_bootstrap_lower(values: dict[str, list[float]], *, seed: int, draws: int = 4096) -> float:
    roots = sorted(values)
    if not roots:
        raise ValueError("bootstrap requires roots")
    root_mean = np.asarray([np.mean(values[root]) for root in roots], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(root_mean), size=(int(draws), len(root_mean)))
    return float(np.quantile(root_mean[indices].mean(axis=1), 0.05))


def _aggregate(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    by_root: dict[str, list[float]] = defaultdict(list)
    for row, utility in zip(rows, policy["utility"]):
        by_root[str(row["instance_id"])].append(float(utility))
    return {
        "decision_group_count": len(rows),
        "root_count": len(by_root),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "route_counts": dict(sorted(Counter(policy["route"]).items())),
        "accuracy": policy["accuracy"], "active_coverage": policy["active_coverage"],
        "active_precision": policy["active_precision"], "support_coverage": policy["support_coverage"],
        "mean_repair_utility": policy["mean_repair_utility"],
        "utility_by_root": {root: float(np.mean(value)) for root, value in sorted(by_root.items())},
    }


def _split_calibration(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    select = [row for row in rows if _fold(str(row["instance_id"])) == 0]
    certify = [row for row in rows if _fold(str(row["instance_id"])) == 1]
    if not select or not certify:
        raise ValueError("calibration roots must populate selector and certificate partitions")
    return select, certify


def _family_support(rows: list[dict[str, Any]], *, minimum_mode_count: int) -> dict[str, dict[str, Any]]:
    """Require calibration examples for each distinct mechanism a family repairs.

    ``metric_repair`` is not one homogeneous intervention: an endpoint-chart
    failure and a task-score failure need different future repair functions.
    A calibration split that has only one cannot authorize the other simply
    because both were previously collapsed into a broad metric label.
    """

    if minimum_mode_count < 1:
        raise ValueError("minimum_mode_count must be positive")
    result = {}
    for family, modes in FAMILY_REQUIRED_MODES.items():
        counts = {
            mode: len({str(row["instance_id"]) for row in rows if row["dominant_failure_mode"] == mode})
            for mode in sorted(modes)
        }
        result[family] = {
            "required_mechanisms": sorted(modes), "root_count_by_mechanism": counts,
            "minimum_root_count_per_mechanism": int(minimum_mode_count),
            "supported": bool(all(value >= minimum_mode_count for value in counts.values())),
        }
    return result


def _fit_feature_support(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a train-only joint-support envelope in standardized feature space."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("support fit requires at least two finite train feature rows")
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (values - mean) / scale
    distance = np.sqrt(np.mean((standardized[:, None, :] - standardized[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(distance, np.inf)
    threshold = float(max(np.quantile(np.min(distance, axis=1), 0.95), 1.0e-8))
    return mean, scale, standardized, threshold


def _feature_support_mask(features: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> np.ndarray:
    mean, scale, train, threshold = fit
    values = (np.asarray(features, dtype=np.float64) - mean) / scale
    distance = np.sqrt(np.mean((values[:, None, :] - train[None, :, :]) ** 2, axis=-1))
    return np.min(distance, axis=1) <= threshold


def run(
    dataset_path: Path, contrast_audit_path: Path, *, seed: int = 202610382, minimum_mode_count: int = 10,
) -> dict[str, Any]:
    dataset = _load_object(dataset_path)
    audit = _load_object(contrast_audit_path)
    if not isinstance(dataset.get("records"), list) or audit.get("schema") != "gdd_wm_teacher_counterfactual_contrast_audit.v0.1":
        raise ValueError("requires candidate dataset and compatible counterfactual contrast audit")
    protocol = audit.get("protocol")
    if not isinstance(protocol, dict) or Path(str(protocol.get("dataset", ""))).resolve() != dataset_path.resolve():
        raise ValueError("contrast audit is not bound to this candidate dataset")
    digest = protocol.get("candidate_dataset_sha256")
    if digest is not None and digest != _sha256(dataset_path):
        raise ValueError("contrast audit dataset SHA256 does not match input")
    audit_rows = {
        (str(row["instance_id"]), str(row["teacher_set_id"])): str(row["dominant_failure_mode"])
        for row in audit.get("per_decision_group", [])
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset["records"]:
        groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(dict(row))
    if set(groups) != set(audit_rows):
        raise ValueError("candidate dataset and contrast audit decision groups differ")
    records: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        root, set_id = key
        split = {str(row["split"]) for row in rows}
        if len(split) != 1 or next(iter(split)) not in {"train", "calibration", "outcome"}:
            raise ValueError(f"{root}/{set_id}: invalid split assignment")
        records.append({
            "instance_id": root, "teacher_set_id": set_id, "split": next(iter(split)),
            "dominant_failure_mode": audit_rows[key], "label": _map_label(audit_rows[key]), "features": _root_features(rows),
        })
    train = [row for row in records if row["split"] == "train"]
    calibration = [row for row in records if row["split"] == "calibration"]
    outcome = [row for row in records if row["split"] == "outcome"]
    if not train or not calibration or not outcome:
        raise ValueError("train/calibration/outcome mechanism groups are all required")
    train_features = np.stack([row["features"] for row in train])
    fit = _fit_centroid_router(train_features, [row["label"] for row in train])
    feature_support = _fit_feature_support(train_features)
    def predict(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
        raw, prob = _predict_centroid(np.stack([row["features"] for row in rows]), fit)
        return raw, np.max(prob, axis=1)

    cal_prediction, cal_confidence = predict(calibration)
    cal_support = _feature_support_mask(np.stack([row["features"] for row in calibration]), feature_support)
    selector, certificate = _split_calibration(calibration)
    support = _family_support(calibration, minimum_mode_count=minimum_mode_count)
    allowed = {family for family, value in support.items() if value["supported"]}
    select_indices = [calibration.index(row) for row in selector]
    certificate_indices = [calibration.index(row) for row in certificate]
    candidates = sorted({0.0, *[float(cal_confidence[index]) for index in select_indices], float("inf")})
    evaluated = []
    for threshold in candidates:
        policy = _policy(
            [row["label"] for row in selector], [cal_prediction[index] for index in select_indices],
            cal_confidence[select_indices], threshold, allowed_families=allowed, support=cal_support[select_indices],
        )
        if policy["active_coverage"] >= 0.10:
            evaluated.append((float(threshold), policy))
    if not evaluated:
        threshold, selected = float("inf"), _policy(
            [row["label"] for row in selector], [cal_prediction[index] for index in select_indices],
            cal_confidence[select_indices], float("inf"), allowed_families=allowed, support=cal_support[select_indices],
        )
    else:
        threshold, selected = max(evaluated, key=lambda item: (item[1]["mean_repair_utility"], item[1]["active_precision"] or -1.0, -item[0]))
    certificate_policy = _policy(
        [row["label"] for row in certificate], [cal_prediction[index] for index in certificate_indices],
        cal_confidence[certificate_indices], threshold, allowed_families=allowed, support=cal_support[certificate_indices],
    )
    cert_summary = _aggregate(certificate, certificate_policy)
    lower = _root_bootstrap_lower(
        {root: [value] for root, value in cert_summary["utility_by_root"].items()}, seed=int(seed) + 31,
    )
    support_transport = bool(float(cal_support.mean()) >= 0.10)
    authorized = bool(
        allowed and support_transport and cert_summary["mean_repair_utility"] > 0.0 and lower > 0.0
        and cert_summary["active_coverage"] >= 0.10
    )
    outcome_prediction, outcome_confidence = predict(outcome)
    outcome_support = _feature_support_mask(np.stack([row["features"] for row in outcome]), feature_support)
    deployed_threshold = threshold if authorized else float("inf")
    outcome_policy = _policy(
        [row["label"] for row in outcome], outcome_prediction, outcome_confidence, deployed_threshold,
        allowed_families=allowed, support=outcome_support,
    )
    diagnostic_policy = _policy(
        [row["label"] for row in outcome], outcome_prediction, outcome_confidence, threshold,
        allowed_families=allowed, support=outcome_support,
    )
    constants = {
        name: _policy([row["label"] for row in outcome], [name] * len(outcome), np.ones(len(outcome)), 0.0, support=outcome_support)
        for name in CLASSES
    }
    if not allowed:
        status = "rejected_calibration_mechanism_support"
    elif not support_transport:
        status = "rejected_train_feature_support_transport"
    elif not authorized:
        status = "rejected_prospective_mechanism_router"
    else:
        status = "authorized_prospective_mechanism_router"
    return {
        "schema": SCHEMA,
        "status": status,
        "candidate_dataset_sha256": _sha256(dataset_path),
        "contrast_audit": str(contrast_audit_path.resolve()),
        "protocol": {
            "fit_labels_from": "train-root post-execution mechanism labels only",
            "threshold_selected_on": "root-disjoint calibration selector partition",
            "authorization_on": "disjoint calibration certificate partition",
            "outcome_labels_read_before_policy_freeze": False,
            "repair_execution_or_planner_selection": False,
            "feature_fields": "predicted endpoint geometry, action-bank geometry, fixed teacher scores, visual history/source latent, goal",
        },
        "router": {"kind": "balanced_standardized_nearest_centroid", "classes": list(CLASSES), "feature_dimension": int(len(train[0]["features"]))},
        "train_feature_support": {
            "kind": "train_only_standardized_nearest_neighbor", "threshold": float(feature_support[3]),
            "calibration_coverage": float(cal_support.mean()), "outcome_coverage": float(outcome_support.mean()),
            "minimum_calibration_coverage": 0.10, "transport_supported": support_transport,
        },
        "calibration": {
            "repair_family_support": support,
            "allowed_repair_families": sorted(allowed),
            "selected_threshold": threshold,
            "selector": _aggregate(selector, selected),
            "certificate": {**cert_summary, "root_bootstrap_5pct_utility": lower, "authorized": authorized},
        },
        "outcome": {
            "deployed": _aggregate(outcome, outcome_policy),
            "diagnostic_if_authorized": _aggregate(outcome, diagnostic_policy),
            "constant_route_baselines": {name: _aggregate(outcome, policy) for name, policy in constants.items()},
        },
        "claim_boundary": "Prospective repair-family routing pilot only. It does not perform metric/dynamics repair, establish action regret improvement, or authorize a planner.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--contrast-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=202610382)
    parser.add_argument("--minimum-mode-count", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite prospective repairability audit: {args.output}")
    report = run(args.dataset, args.contrast_audit, seed=args.seed, minimum_mode_count=args.minimum_mode_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "calibration": report["calibration"]["certificate"], "outcome": report["outcome"]["deployed"]}, indent=2, sort_keys=True))
    return 0 if report["status"].startswith("authorized") else 2


if __name__ == "__main__":
    raise SystemExit(main())
