"""Evaluate frozen V8 PointMaze arms with a fixed-bank HTSC release rule.

The V8 shared-future study is a three-arm *offline* control.  This evaluator
is deliberately separate from fitting: it reads the already-frozen candidate
tables, calibrates only on V8 calibration roots, then applies the same
fixed-bank permit/probe rule to disjoint V8 outcome roots.  It is not an
online-planning guarantee, and its fallback is explicitly the frozen direct
cost-regression arm rather than a privileged controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    calibrate_root_score_bound,
    certify_candidate_bank,
    evaluate_contrast_bound,
)


SCHEMA = "pointmaze_shared_future_v8_htsc_evaluation.v1"
REQUIRED_ARMS = frozenset({"cost_regression", "planning_shaped", "shared_future_planning"})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_action(value: Any) -> str:
    action = np.asarray(value, dtype=np.float64)
    if action.shape != (20, 2) or not np.isfinite(action).all():
        raise ValueError("each V8 candidate action must be finite [20,2]")
    return json.dumps(action.round(12).tolist(), separators=(",", ":"))


def _arms(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = report.get("candidate_objectives")
    if not isinstance(entries, list):
        raise ValueError("V8 report is missing candidate_objectives")
    arms: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("candidate_objectives must contain named objects")
        name = str(entry["name"])
        if name in arms:
            raise ValueError(f"duplicate candidate objective {name!r}")
        arms[name] = entry
    missing = sorted(REQUIRED_ARMS - set(arms))
    if missing:
        raise ValueError(f"V8 HTSC requires frozen arms: {missing}")
    return arms


def _validate_v8_contract(
    report: dict[str, Any], preflight: dict[str, Any], *, expected_preflight_sha256: str | None = None,
) -> None:
    if report.get("schema") != "pointmaze_planning_shaped_execution_control.v0.4":
        raise ValueError("unsupported planning-shaped control report schema")
    contract = report.get("control_contract")
    if not isinstance(contract, dict):
        raise ValueError("V8 report is missing its control contract")
    if contract.get("action_representation") != "full_ordered_20x2":
        raise ValueError("V8 HTSC requires full_ordered_20x2 actions")
    if contract.get("shared_future_control_included") is not True:
        raise ValueError("V8 report did not run the action-independent shared-future arm")
    receipt = contract.get("v8_static_preflight")
    if not isinstance(receipt, dict) or receipt.get("verified") is not True:
        raise ValueError("V8 report lacks an execution-boundary static-preflight verification")
    if receipt.get("root_counts") != {"train": 128, "calibration": 192, "outcome": 64}:
        raise ValueError("V8 execution-boundary preflight root counts are invalid")
    if receipt.get("action_representation") != "full_ordered_20x2":
        raise ValueError("V8 execution-boundary preflight action representation is invalid")
    if expected_preflight_sha256 is not None and receipt.get("ledger_sha256") != expected_preflight_sha256:
        raise ValueError("V8 control report was executed against a different static preflight ledger")
    split = report.get("split_isolation")
    expected = {"train_root_count": 128, "calibration_root_count": 192, "outcome_root_count": 64}
    if not isinstance(split, dict) or any(int(split.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("V8 report does not carry the frozen 128/192/64 root split")
    if split.get("root_disjoint") is not True:
        raise ValueError("V8 report does not declare root-disjoint splits")
    if preflight.get("schema") != "pointmaze_shared_future_v8_static_preflight.v1":
        raise ValueError("unsupported V8 preflight schema")
    if preflight.get("status") != "passed_static_preflight":
        raise ValueError("V8 static preflight was not passed")
    bank = preflight.get("candidate_bank")
    if not isinstance(bank, dict) or bank.get("shared_candidate_in_support") is not True:
        raise ValueError("V8 candidate bank is not declared in support")
    if bank.get("scorer_candidate_bank_protocol") != "held_exact_shared_bank":
        raise ValueError("V8 scorer candidate bank contract is not exact shared support")


def _candidate_table(arm: dict[str, Any], field: str) -> dict[str, list[dict[str, Any]]]:
    rows = arm.get(field)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"arm lacks non-empty {field}")
    table: dict[str, list[dict[str, Any]]] = {}
    expected_ids: tuple[str, ...] | None = None
    action_by_id: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
            raise ValueError(f"{field} must contain root-level candidate rows")
        root = str(row["instance_id"])
        if root in table:
            raise ValueError(f"duplicate root {root!r} in {field}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"{root}: requires at least two candidates")
        copied: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"{root}: candidate must be an object")
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ValueError(f"{root}: candidate_id must be a string")
            predicted = float(candidate.get("predicted_cost"))
            realized = float(candidate.get("realized_cost"))
            if not np.isfinite(predicted) or not np.isfinite(realized) or realized < 0.0:
                raise ValueError(f"{root}/{candidate_id}: invalid candidate costs")
            action = _canonical_action(candidate.get("action_sequence"))
            prior = action_by_id.setdefault(candidate_id, action)
            if prior != action:
                raise ValueError("all V8 roots must use one exact action sequence per candidate ID")
            copied.append({
                "candidate_id": candidate_id, "predicted_cost": predicted,
                "realized_cost": realized, "action": action,
            })
        copied.sort(key=lambda value: value["candidate_id"])
        ids = tuple(value["candidate_id"] for value in copied)
        if len(ids) != len(set(ids)):
            raise ValueError(f"{root}: candidate IDs must be unique")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("all V8 roots must share identical candidate IDs")
        table[root] = copied
    return table


def _risk_score(rows: list[dict[str, Any]]) -> float:
    """A model-only ambiguity score fixed before outcome costs are read."""

    scores = np.sort(np.asarray([row["predicted_cost"] for row in rows], dtype=np.float64))
    return float(1.0 / max(float(scores[1] - scores[0]), 1.0e-12))


def _winner(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return min(rows, key=lambda row: (float(row[field]), str(row["candidate_id"])))


def _flatten(table: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual, predicted, roots = [], [], []
    for root, rows in sorted(table.items()):
        actual.extend(float(row["realized_cost"]) for row in rows)
        predicted.extend(float(row["predicted_cost"]) for row in rows)
        roots.extend([root] * len(rows))
    return np.asarray(actual), np.asarray(predicted), np.asarray(roots, dtype=object)


def _evaluate_arm(
    name: str, calibration: dict[str, list[dict[str, Any]]], outcome: dict[str, list[dict[str, Any]]],
    fallback_outcome: dict[str, list[dict[str, Any]]], *, alpha: float, risk_quantile: float,
) -> dict[str, Any]:
    if set(calibration) & set(outcome):
        raise ValueError("calibration and outcome roots must be disjoint")
    if set(outcome) != set(fallback_outcome):
        raise ValueError("fallback arm must cover exactly the outcome roots")
    real, predicted, groups = _flatten(calibration)
    contrast = calibrate_contrast_bound(real, predicted, groups, sorted(calibration), alpha=alpha)
    risk = calibrate_root_score_bound(
        np.asarray([_risk_score(rows) for _, rows in sorted(calibration.items())]),
        alpha=float(1.0 - risk_quantile),
    )
    bound = contrast["bound"]
    threshold = risk["bound"]
    choices: list[dict[str, Any]] = []
    # Every status and selected candidate is fixed from model scores,
    # calibration-only thresholds, and the declared support contract.  Only
    # then are realized costs attached as outcome diagnostics below.
    for root, rows in sorted(outcome.items()):
        fallback = _winner(fallback_outcome[root], "predicted_cost")
        model_choice = _winner(rows, "predicted_cost")
        certificate = certify_candidate_bank(
            np.asarray([row["predicted_cost"] for row in rows]), bound,
            support_passed=True, candidate_ids=[row["candidate_id"] for row in rows],
            candidate_bank_mode="fixed",
        )
        current_risk = _risk_score(rows)
        risk_passed = bool(threshold is not None and current_risk <= float(threshold))
        if not risk_passed:
            status, reason = "probe", "calibration_only_model_ambiguity_gate_failed"
        else:
            status, reason = str(certificate["status"]), str(certificate["reason"])
        chosen = model_choice if status == "permit" else fallback
        choices.append({
            "root_id": root, "status": status, "reason": reason,
            "model_candidate_id": model_choice["candidate_id"],
            "fallback_candidate_id": fallback["candidate_id"],
            "selected_candidate_id": chosen["candidate_id"],
            "risk_score": current_risk, "risk_threshold": threshold,
            "minimum_predicted_margin": certificate["minimum_predicted_margin"],
        })
    outcome_real, outcome_predicted, outcome_groups = _flatten(outcome)
    coverage = evaluate_contrast_bound(outcome_real, outcome_predicted, outcome_groups, bound)
    rows_by_id = {root: {row["candidate_id"]: row for row in values} for root, values in outcome.items()}
    fallback_by_id = {root: {row["candidate_id"]: row for row in values} for root, values in fallback_outcome.items()}
    released, wrong_releases, gains, selected_costs, fallback_costs = 0, 0, [], [], []
    for row in choices:
        root = str(row["root_id"])
        oracle = _winner(outcome[root], "realized_cost")
        selected_cost = float(rows_by_id[root][str(row["selected_candidate_id"])]["realized_cost"])
        fallback_cost = float(fallback_by_id[root][str(row["fallback_candidate_id"])]["realized_cost"])
        model_wrong = str(row["model_candidate_id"]) != str(oracle["candidate_id"])
        permitted = row["status"] == "permit"
        released += int(permitted)
        wrong_releases += int(permitted and model_wrong)
        row.update({
            "oracle_candidate_id": oracle["candidate_id"],
            "posthoc_model_ranking_failure": bool(model_wrong),
            "posthoc_selected_regret": float(selected_cost - float(oracle["realized_cost"])),
            "posthoc_fallback_regret": float(fallback_cost - float(oracle["realized_cost"])),
            "posthoc_utility_gain_over_fallback": float(fallback_cost - selected_cost),
        })
        gains.append(float(fallback_cost - selected_cost))
        selected_costs.append(selected_cost)
        fallback_costs.append(fallback_cost)
    return {
        "arm": name, "calibration": {"contrast": contrast, "risk": risk, "risk_quantile": float(risk_quantile)},
        "outcome": {
            "root_count": len(choices), "released_root_count": released,
            "release_rate": float(released / len(choices)),
            "wrong_release_count": wrong_releases,
            "wrong_release_rate": None if released == 0 else float(wrong_releases / released),
            "mean_selected_realized_cost": float(np.mean(selected_costs)),
            "mean_fallback_realized_cost": float(np.mean(fallback_costs)),
            "mean_utility_gain_over_fallback": float(np.mean(gains)),
            "conformal_outcome_coverage": coverage, "roots": choices,
        },
        "claim_boundary": "V8 fixed-bank, calibration-frozen marginal release audit; neither online planning nor permit-conditional/state-conditional coverage.",
    }


def _validate_cross_arm_tables(
    tables: dict[str, dict[str, list[dict[str, Any]]]], *, field: str,
) -> None:
    """Require the three frozen arms to face identical realized decisions."""

    reference = tables["cost_regression"]
    for name, table in tables.items():
        if set(table) != set(reference):
            raise ValueError(f"{name}: {field} roots differ from cost_regression")
        for root in sorted(reference):
            left, right = reference[root], table[root]
            if len(left) != len(right):
                raise ValueError(f"{name}/{root}: candidate count differs across arms")
            for baseline, candidate in zip(left, right, strict=True):
                if baseline["candidate_id"] != candidate["candidate_id"] or baseline["action"] != candidate["action"]:
                    raise ValueError(f"{name}/{root}: candidate bank differs across arms")
                if not np.isclose(baseline["realized_cost"], candidate["realized_cost"], rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"{name}/{root}: realized candidate costs differ across arms")


def evaluate(
    report: dict[str, Any], preflight: dict[str, Any], *, alpha: float = 0.1,
    risk_quantile: float = 0.9, expected_preflight_sha256: str | None = None,
) -> dict[str, Any]:
    if not 0.0 < float(alpha) < 1.0 or not 0.0 < float(risk_quantile) < 1.0:
        raise ValueError("alpha and risk_quantile must lie strictly between zero and one")
    _validate_v8_contract(report, preflight, expected_preflight_sha256=expected_preflight_sha256)
    arms = _arms(report)
    calibration = {name: _candidate_table(entry, "calibration_candidate_costs") for name, entry in arms.items()}
    outcome = {name: _candidate_table(entry, "outcome_candidate_costs") for name, entry in arms.items()}
    _validate_cross_arm_tables(calibration, field="calibration_candidate_costs")
    _validate_cross_arm_tables(outcome, field="outcome_candidate_costs")
    evaluations = {
        name: _evaluate_arm(name, calibration[name], outcome[name], outcome["cost_regression"], alpha=alpha, risk_quantile=risk_quantile)
        for name in sorted(REQUIRED_ARMS)
    }
    return {
        "schema": SCHEMA, "status": "completed_frozen_v8_htsc_audit",
        "protocol": {
            "fallback_arm": "cost_regression", "support_gate": "passed by V8 exact shared candidate-in-support preflight",
            "risk_gate": "calibration-only inverse predicted top-two score gap", "candidate_bank": "fixed exact sequences",
            "outcome_used_for_selection": False, "arms": sorted(REQUIRED_ARMS),
        },
        "arm_evaluations": evaluations,
        "claim_boundary": "Independent V8 fixed-bank HTSC audit of frozen visual-history control arms; it does not establish a universal world model, online visual planning, or conditional safety coverage.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--risk-quantile", type=float, default=0.9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = evaluate(
        _load(args.report), _load(args.preflight), alpha=args.alpha, risk_quantile=args.risk_quantile,
        expected_preflight_sha256=_sha256(args.preflight),
    )
    result["provenance"] = {
        "control_report": str(args.report.resolve()), "control_report_sha256": _sha256(args.report),
        "static_preflight": str(args.preflight.resolve()), "static_preflight_sha256": _sha256(args.preflight),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "release_counts": {name: value["outcome"]["released_root_count"] for name, value in result["arm_evaluations"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
