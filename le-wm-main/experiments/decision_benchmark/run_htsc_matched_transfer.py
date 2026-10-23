"""Evaluate a frozen, fixed-bank HTSC decision-transfer protocol.

This module is deliberately an *offline* decision-transfer evaluator.  It
freezes a candidate bank and a world-model score before calibration, fits the
conformal contrast envelope and a model-only risk threshold on calibration
roots, then evaluates a safe HTSC policy against a coverage-matched fallback
on disjoint test roots.  Test task costs and successes are read only after
both policies have selected their candidate.

The result is not a conditional conformal guarantee: the contrast certificate
is a marginal fixed-bank guarantee under exchangeable roots.  The bundled
PointMaze adapter is a smoke-only bridge because its fallback and support
proxy use evaluator state fields that are unavailable to an RGB controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    calibrate_root_score_bound,
    certify_candidate_bank,
    certify_candidate_bank_regret_budget,
    evaluate_contrast_bound,
)


_SPLITS = frozenset({"train", "calibration", "test"})
_REQUIRED_RECORD_FIELDS = frozenset({
    "root_id", "split", "candidate_id", "action_sequence", "wm_score",
    "fallback_score", "true_cost", "success", "risk_score", "support_passed",
})


def _canonical_action(value: Any) -> str:
    """Return a stable exact representation for a declared candidate action."""

    action = np.asarray(value, dtype=np.float64)
    if action.ndim < 1 or not np.isfinite(action).all():
        raise ValueError("action_sequence must contain finite numeric values")
    return json.dumps(action.round(12).tolist(), separators=(",", ":"))


def _root_groups(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["root_id"])].append(record)
    return {root: sorted(rows, key=lambda row: int(row["candidate_id"])) for root, rows in groups.items()}


def _validate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not records:
        raise ValueError("at least one record is required")
    for index, record in enumerate(records):
        missing = sorted(_REQUIRED_RECORD_FIELDS - set(record))
        if missing:
            raise ValueError(f"record {index} is missing fields: {missing}")
        if str(record["split"]) not in _SPLITS:
            raise ValueError("split must be one of train, calibration, test")
        for field in ("wm_score", "fallback_score", "true_cost", "risk_score"):
            if not np.isfinite(float(record[field])):
                raise ValueError(f"{field} must be finite")
        if float(record["true_cost"]) < 0.0 or float(record["risk_score"]) < 0.0:
            raise ValueError("true_cost and risk_score must be non-negative")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {split: _root_groups([]) for split in _SPLITS}
    root_split: dict[str, str] = {}
    expected_ids: tuple[int, ...] | None = None
    action_by_id: dict[int, str] = {}
    for record in records:
        root = str(record["root_id"])
        split = str(record["split"])
        prior = root_split.setdefault(root, split)
        if prior != split:
            raise ValueError(f"root {root!r} appears in both {prior!r} and {split!r}")
        grouped[split].setdefault(root, []).append(record)
    if not grouped["train"] or not grouped["calibration"] or not grouped["test"]:
        raise ValueError("frozen protocol requires non-empty train, calibration, and test roots")

    for split_groups in grouped.values():
        for root, rows in split_groups.items():
            rows.sort(key=lambda row: int(row["candidate_id"]))
            ids = tuple(int(row["candidate_id"]) for row in rows)
            if len(ids) < 2 or len(set(ids)) != len(ids):
                raise ValueError(f"root {root!r} must contain at least two unique candidates")
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError("all roots must share identical declared candidate IDs")
            root_risks = {float(row["risk_score"]) for row in rows}
            root_support = {bool(row["support_passed"]) for row in rows}
            if len(root_risks) != 1 or len(root_support) != 1:
                raise ValueError("risk_score and support_passed must be root-level quantities")
            for row in rows:
                candidate = int(row["candidate_id"])
                action = _canonical_action(row["action_sequence"])
                reference = action_by_id.setdefault(candidate, action)
                if reference != action:
                    raise ValueError(
                        "all roots must use one fixed action sequence for each candidate ID"
                    )
    return grouped


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("risk_quantile must lie strictly between zero and one")
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        return float(np.quantile(values, quantile, interpolation="higher"))


def _bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0, "roots": 0}
    if values.size == 1:
        value = float(values[0])
        return {"estimate": value, "lower": value, "upper": value, "roots": 1}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "roots": int(values.size),
    }


def _winner(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return min(rows, key=lambda row: (float(row[field]), int(row["candidate_id"])))


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    alpha: float = 0.1,
    risk_quantile: float = 0.9,
    bootstrap_draws: int = 10_000,
    seed: int = 20261021,
    certificate_kind: str = "unique_winner",
    regret_budget: float | None = None,
) -> dict[str, Any]:
    """Evaluate frozen HTSC and its identical-coverage fallback.

    ``true_cost`` and ``success`` are intentionally touched only in calibration
    contrast scoring and test reporting.  Candidate selections on test roots
    depend only on frozen scores, calibration thresholds, and root support.
    """

    if certificate_kind not in {"unique_winner", "regret_budget"}:
        raise ValueError("certificate_kind must be 'unique_winner' or 'regret_budget'")
    if certificate_kind == "regret_budget":
        if regret_budget is None or not np.isfinite(float(regret_budget)) or float(regret_budget) < 0.0:
            raise ValueError("regret_budget certificate requires a finite non-negative regret_budget")
    elif regret_budget is not None:
        raise ValueError("regret_budget is only valid with certificate_kind='regret_budget'")
    grouped = _validate_records(records)
    calibration_rows = [row for rows in grouped["calibration"].values() for row in rows]
    calibration_groups = np.asarray([
        row["root_id"] for row in calibration_rows
    ], dtype=object)
    contrast = calibrate_contrast_bound(
        np.asarray([float(row["true_cost"]) for row in calibration_rows]),
        np.asarray([float(row["wm_score"]) for row in calibration_rows]),
        calibration_groups,
        sorted(grouped["calibration"]),
        alpha=alpha,
    )
    calibration_risks = np.asarray([
        float(rows[0]["risk_score"]) for _, rows in sorted(grouped["calibration"].items())
    ])
    # A heuristic empirical quantile cannot be described as finite-sample
    # calibrated.  Use the same split-conformal rank convention as the
    # contrast envelope and fail closed when this calibration panel is too
    # small for the requested quantile.
    risk_calibration = calibrate_root_score_bound(
        calibration_risks, alpha=float(1.0 - risk_quantile),
    )
    risk_threshold = risk_calibration["bound"]

    test_rows = []
    for root, rows in sorted(grouped["test"].items()):
        wm_choice = _winner(rows, "wm_score")
        fallback_choice = _winner(rows, "fallback_score")
        root_risk = float(rows[0]["risk_score"])
        support_passed = bool(rows[0]["support_passed"])
        risk_passed = bool(risk_threshold is not None and root_risk <= float(risk_threshold))
        # Keep the certificate independent of the support status as well as
        # recording the deployment status below.  Previously the single
        # ``reason`` field silently masked a simultaneous margin/coverage
        # failure whenever the risk gate failed (and masked both whenever
        # support failed).  That made a zero-release panel impossible to
        # diagnose: a root has a vector of gate failures, not one exclusive
        # cause.  The decision itself is unchanged and still fail-closed.
        margin_certificate = certify_candidate_bank(
            np.asarray([float(row["wm_score"]) for row in rows]),
            contrast["bound"],
            support_passed=True,
            candidate_ids=[int(row["candidate_id"]) for row in rows],
            candidate_bank_mode="fixed",
        )
        certificate_input = np.asarray([float(row["wm_score"]) for row in rows])
        if certificate_kind == "regret_budget":
            certificate = certify_candidate_bank_regret_budget(
                certificate_input, contrast["bound"], regret_budget=float(regret_budget),
                support_passed=support_passed,
                candidate_ids=[int(row["candidate_id"]) for row in rows],
                candidate_bank_mode="fixed",
            )
        else:
            certificate = certify_candidate_bank(
                certificate_input, contrast["bound"],
                support_passed=support_passed,
                candidate_ids=[int(row["candidate_id"]) for row in rows],
                candidate_bank_mode="fixed",
            )
        failure_vector = {
            "support_failure": not support_passed,
            "uncertainty_failure": not risk_passed,
            # This is observable before execution.  It is deliberately not
            # named a coverage failure: coverage is an outcome-only property
            # and cannot legally route a test-time action.
            "certificate_margin_failure": str(margin_certificate["status"]) != "permit",
        }
        if not risk_passed and support_passed:
            status = "probe"
            reason = "calibration_only_model_risk_gate_failed"
        else:
            status = str(certificate["status"])
            reason = str(certificate["reason"])
        permit = status == "permit"
        htsc_choice = wm_choice if permit else fallback_choice
        oracle = _winner(rows, "true_cost")
        test_rows.append({
            "root_id": root,
            "status": status,
            "reason": reason,
            "support_passed": support_passed,
            "risk_score": root_risk,
            "risk_threshold": risk_threshold,
            "risk_passed": risk_passed,
            "contrast_bound": contrast["bound"],
            "minimum_predicted_margin": certificate["minimum_predicted_margin"],
            "gate_failure_vector": failure_vector,
            "margin_certificate_status": str(margin_certificate["status"]),
            "margin_certificate_reason": str(margin_certificate["reason"]),
            "htsc_candidate_id": int(htsc_choice["candidate_id"]),
            "matched_fallback_candidate_id": int(fallback_choice["candidate_id"]),
            "oracle_candidate_id": int(oracle["candidate_id"]),
            "htsc_regret": float(htsc_choice["true_cost"] - oracle["true_cost"]),
            "matched_fallback_regret": float(fallback_choice["true_cost"] - oracle["true_cost"]),
            "htsc_success": float(bool(htsc_choice["success"])),
            "matched_fallback_success": float(bool(fallback_choice["success"])),
            "paired_regret_gain": float(fallback_choice["true_cost"] - htsc_choice["true_cost"]),
            "paired_success_gain": float(bool(htsc_choice["success"])) - float(bool(fallback_choice["success"])),
            # This is an outcome-only post-hoc diagnostic.  It is deliberately
            # added after the action choice and cannot participate in gates or
            # threshold calibration.
            "posthoc_utility_ranking_failure": bool(int(wm_choice["candidate_id"]) != int(oracle["candidate_id"])),
        })

    # Keep the coverage call on candidate-level records, rather than selected
    # roots, so it remains the simultaneous fixed-bank conformal diagnostic.
    test_candidate_rows = [row for rows in grouped["test"].values() for row in rows]
    coverage = evaluate_contrast_bound(
        np.asarray([float(row["true_cost"]) for row in test_candidate_rows]),
        np.asarray([float(row["wm_score"]) for row in test_candidate_rows]),
        np.asarray([str(row["root_id"]) for row in test_candidate_rows]),
        contrast["bound"],
    )
    coverage_by_root = {str(row["group_id"]): row["covered"] for row in coverage["roots"]}
    for row in test_rows:
        # Outcome-only diagnosis, written after the frozen gate and candidate
        # choice.  It must never feed the release/recovery policy above.
        row["posthoc_contrast_covered"] = coverage_by_root[str(row["root_id"])]
        row["posthoc_coverage_failure"] = not bool(row["posthoc_contrast_covered"])
    regret_gain = np.asarray([float(row["paired_regret_gain"]) for row in test_rows])
    success_gain = np.asarray([float(row["paired_success_gain"]) for row in test_rows])
    released = [row for row in test_rows if row["status"] == "permit"]
    failure_counts = {
        key: int(sum(bool(row["gate_failure_vector"][key]) for row in test_rows))
        for key in ("support_failure", "uncertainty_failure", "certificate_margin_failure")
    }
    posthoc_coverage_failure_count = int(sum(bool(row["posthoc_coverage_failure"]) for row in test_rows))
    utility_ranking_failures = int(sum(bool(row["posthoc_utility_ranking_failure"]) for row in test_rows))
    return {
        "schema": "htsc_matched_transfer_evaluation.v1",
        "protocol": {
            "train_calibration_test_root_disjoint": True,
            "candidate_bank": "one exact action sequence per candidate ID across every root",
            "candidate_generation": "fixed before calibration and test scoring",
            "certificate_kind": certificate_kind,
            "regret_budget": None if regret_budget is None else float(regret_budget),
            "risk_gate": "root-level model-only risk threshold fitted on calibration roots",
            "matched_fallback": "uses the exact HTSC release mask; differs only on permitted roots",
            "test_outcome_used_for_selection": False,
            "conformal_scope": "marginal fixed-bank root-level contrast coverage under exchangeability",
        },
        "calibration": {
            "contrast": contrast,
            "risk_quantile": float(risk_quantile),
        "risk_threshold": risk_threshold,
        "risk_calibration": risk_calibration,
            "risk_calibration_root_count": int(calibration_risks.size),
        },
        "test": {
            "root_count": int(len(test_rows)),
            "released_root_count": int(len(released)),
            "release_rate": float(len(released) / len(test_rows)),
            "gate_failure_counts": failure_counts,
            "posthoc_coverage_failure_count": posthoc_coverage_failure_count,
            "posthoc_utility_ranking_failure_count": utility_ranking_failures,
            "matched_fallback_root_count": int(len(test_rows)),
            "paired_regret_gain_mean": float(regret_gain.mean()),
            "paired_success_gain_mean": float(success_gain.mean()),
            "paired_regret_gain_bootstrap": _bootstrap_mean(regret_gain, draws=bootstrap_draws, seed=seed),
            "paired_success_gain_bootstrap": _bootstrap_mean(success_gain, draws=bootstrap_draws, seed=seed + 1),
            "conformal_outcome_coverage": coverage,
            "roots": test_rows,
        },
        "claim_boundary": (
            "Offline fixed-bank matched-fallback decision-transfer evaluation. "
            "The conformal component is marginal rather than permit-conditional, "
            "and physical regret/success are independent held-out consequences rather "
            "than calibrated physical-cost guarantees."
        ),
    }


def _pointmaze_records(dataset: Path, rgb_report: Path, fit_export: Path, *, max_test_roots: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a smoke table from an existing PointMaze RGB successor artifact."""

    from experiments.decision_benchmark.audit_rgb_task_contrast import _root_metrics
    from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _fit_readouts

    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = list(payload["records"])
    rgb_payload = json.loads(rgb_report.read_text(encoding="utf-8"))
    rgb_rows = {(str(row["instance_id"]), int(row["candidate_id"])): row for row in rgb_payload["rows"]}
    fit_payload = load_payload(fit_export)
    metrics = _root_metrics(rows, rgb_rows, _fit_readouts(fit_payload["enriched_records"])[4])
    metric_by_key = {(str(row["instance_id"]), int(row["candidate_id"])): row for row in metrics}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {} for split in _SPLITS
    }
    for row in rows:
        split = "test" if str(row["split"]) == "outcome" else str(row["split"])
        root = str(row["instance_id"])
        grouped[split].setdefault(root, []).append({
            "root_id": root,
            "split": split,
            "candidate_id": int(row["candidate_id"]),
            "action_sequence": row["action_sequence"],
            "row": row,
        })
    for split_groups in grouped.values():
        for root_rows in split_groups.values():
            root_rows.sort(key=lambda item: int(item["candidate_id"]))
    test_roots = sorted(grouped["test"])
    if max_test_roots is not None:
        test_roots = test_roots[: int(max_test_roots)]
    selected_test = set(test_roots)
    train_sources = np.asarray([
        np.asarray(root_rows[0]["row"]["source_state_raw"], dtype=np.float64)
        for root_rows in grouped["train"].values()
    ])
    mean = train_sources.mean(axis=0)
    scale = np.maximum(train_sources.std(axis=0), 1.0e-6)
    train_distances = np.sqrt(np.mean(((train_sources - mean) / scale) ** 2, axis=1))
    support_threshold = _higher_quantile(train_distances, 0.9)
    output: list[dict[str, Any]] = []
    for split, roots in grouped.items():
        for root, root_rows in roots.items():
            if split == "test" and root not in selected_test:
                continue
            wm_scores = np.asarray([
                float(metric_by_key[(root, int(item["candidate_id"]))]["predicted_cost"])
                for item in root_rows
            ])
            ordered = np.sort(wm_scores)
            margin = float(ordered[1] - ordered[0])
            risk = float(1.0 / max(margin, 1.0e-6))
            source = np.asarray(root_rows[0]["row"]["source_state_raw"], dtype=np.float64)[:2]
            goal = np.asarray(root_rows[0]["row"]["goal_raw"], dtype=np.float64)[:2]
            direction = goal - source
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            support_distance = float(np.sqrt(np.mean(
                ((np.asarray(root_rows[0]["row"]["source_state_raw"], dtype=np.float64) - mean) / scale) ** 2
            )))
            support_passed = bool(support_distance <= support_threshold)
            for item in root_rows:
                row = item["row"]
                metric = metric_by_key[(root, int(item["candidate_id"]))]
                action_sum = np.asarray(row["action_sequence"], dtype=np.float64).sum(axis=0)
                output.append({
                    "root_id": root,
                    "split": split,
                    "candidate_id": int(item["candidate_id"]),
                    "action_sequence": row["action_sequence"],
                    "wm_score": float(metric["predicted_cost"]),
                    "fallback_score": float(-action_sum @ direction),
                    "true_cost": float(metric["real_task_cost"]),
                    "success": bool(float(metric["real_task_cost"]) <= 0.25),
                    "risk_score": risk,
                    "support_passed": support_passed,
                })
    return output, {
        "dataset": str(dataset.resolve()),
        "rgb_report": str(rgb_report.resolve()),
        "fit_export": str(fit_export.resolve()),
        "fallback": "offline action-goal alignment proxy using evaluator source/goal coordinates",
        "support": "offline train-root source-state distance proxy",
        "selected_test_roots": sorted(selected_test),
    }


def _cuda_probe(device: str) -> dict[str, Any]:
    if device == "cpu":
        return {"device": "cpu", "status": "not_requested"}
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing a CPU fallback")
    tensor = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    return {
        "device": "cuda",
        "status": "passed",
        "name": torch.cuda.get_device_name(0),
        "allocation_sum": float(tensor.sum().item()),
        "torch_cuda": str(torch.version.cuda),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--risk-quantile", type=float, default=0.9)
    parser.add_argument("--max-test-roots", type=int, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = _cuda_probe(args.device)
    records, provenance = _pointmaze_records(
        args.dataset, args.rgb_report, args.fit_export, max_test_roots=args.max_test_roots,
    )
    report = evaluate_records(
        records,
        alpha=args.alpha,
        risk_quantile=args.risk_quantile,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    report["runtime"] = {"device": device, "pointmaze_smoke_provenance": provenance}
    report["claim_boundary"] += (
        " This PointMaze convenience entry uses evaluator state only to construct an "
        "offline fallback/support proxy and is therefore a protocol smoke, not RGB-only planning evidence."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "released_root_count": report["test"]["released_root_count"],
        "paired_regret_gain_mean": report["test"]["paired_regret_gain_mean"],
        "paired_success_gain_mean": report["test"]["paired_success_gain_mean"],
        "device": device,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
