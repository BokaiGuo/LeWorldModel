"""Audit a sealed prospective PointMaze repairability comparison matrix.

This is deliberately an *outcome-only* component.  The runner which creates
``decision.json`` must freeze every scorer and every root-level route before
the executor creates ``outcomes.json``.  Keeping the two artifacts separate
means the audit cannot quietly tune a threshold using held-out task costs.

The required arms are a matched action-effect carrier, uniform history repair,
ordinary task-preference learning, and a diagnostic-selected route.  Four
simple diagnostic routes (visible motion, predictive MSE, support, and ATM)
are controls rather than optional descriptive columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze_prospective_repairability_matrix_audit.v1"
DECISION_SCHEMA = "pointmaze_prospective_repairability_matrix_decision.v1"
OUTCOME_SCHEMA = "pointmaze_prospective_repairability_matrix_outcomes.v1"
ARMS = ("action_effect_only", "uniform_history_repair", "ordinary_preference")
POLICIES = (
    "action_effect_only",
    "uniform_history_repair",
    "ordinary_preference",
    "repairability_diagnostic",
    "visible_motion_control",
    "mse_control",
    "support_control",
    "atm_control",
)
_EPS = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _root_key(row: dict[str, Any]) -> str:
    value = row.get("root_id", row.get("instance_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("each decision row needs a nonempty root_id")
    return value


def _candidate_key(row: dict[str, Any]) -> str:
    root = _root_key(row)
    try:
        candidate = int(row["candidate_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("each decision row needs an integer candidate_id") from error
    return f"{root}/{candidate}"


def _root_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_root_key(row)].append(row)
    if not groups:
        raise ValueError("decision artifact has no rows")
    for root, values in groups.items():
        values.sort(key=lambda row: int(row["candidate_id"]))
        ids = [int(row["candidate_id"]) for row in values]
        if ids != list(range(len(ids))):
            raise ValueError(f"root {root!r} has non-contiguous candidate ids")
        split = {str(row.get("split")) for row in values}
        if len(split) != 1 or split.pop() != "outcome":
            raise ValueError("matrix audit accepts outcome roots only")
    return dict(sorted(groups.items()))


def _validate_decision(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if payload.get("schema") != DECISION_SCHEMA:
        raise ValueError("unexpected decision schema")
    if payload.get("outcome_labels_opened") is not False:
        raise ValueError("decision artifact was not sealed before outcome labels")
    rows, routes = payload.get("records"), payload.get("root_routes")
    if not isinstance(rows, list) or not isinstance(routes, dict):
        raise ValueError("decision artifact needs records and root_routes")
    if set(routes) != set(POLICIES):
        raise ValueError("decision artifact lacks a required policy route")
    for row in rows:
        if not isinstance(row, dict) or "true_cost" in row or "realized_terminal_state" in row:
            raise ValueError("decision rows must not contain held-out outcome labels")
        scores = row.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(ARMS):
            raise ValueError("each decision row needs exactly the three preregistered arm scores")
        if not all(np.isfinite(float(scores[name])) for name in ARMS):
            raise ValueError("decision arm scores must be finite")
    groups = _root_groups(rows)
    roots = set(groups)
    normalized: dict[str, str] = {}
    for policy, values in routes.items():
        if not isinstance(values, dict) or set(values) != roots:
            raise ValueError(f"policy {policy!r} does not cover exactly the outcome roots")
        for root, arm in values.items():
            if arm not in ARMS:
                raise ValueError(f"policy {policy!r} selected unknown arm {arm!r}")
        normalized[policy] = "ok"
    return rows, normalized


def _validate_outcomes(payload: dict[str, Any], decision_path: Path, rows: list[dict[str, Any]]) -> dict[str, float]:
    if payload.get("schema") != OUTCOME_SCHEMA:
        raise ValueError("unexpected outcome schema")
    if payload.get("decision_sha256") != _sha256(decision_path):
        raise ValueError("outcomes are not hash-bound to the sealed decision artifact")
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("outcomes need a labels object")
    expected = {_candidate_key(row) for row in rows}
    if set(labels) != expected:
        raise ValueError("outcome labels do not exactly cover the sealed candidate rows")
    values = {key: float(value["real_task_cost"] if isinstance(value, dict) else value) for key, value in labels.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("outcome costs must be finite")
    return values


def _selection(groups: dict[str, list[dict[str, Any]]], labels: dict[str, float], routes: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {name: {} for name in POLICIES}
    for root, rows in groups.items():
        actual = np.asarray([labels[_candidate_key(row)] for row in rows], dtype=np.float64)
        oracle = float(actual.min())
        for policy in POLICIES:
            arm = str(routes[policy][root])
            score = np.asarray([float(row["scores"][arm]) for row in rows], dtype=np.float64)
            winner = int(np.argmin(score))
            result[policy][root] = {
                "selected_arm": arm,
                "winner_candidate_id": int(rows[winner]["candidate_id"]),
                "regret": float(actual[winner] - oracle),
                "ranking_correct": bool(winner == int(np.argmin(actual))),
                "success": bool(actual[winner] <= 0.25),
            }
    return result


def _bootstrap(values: np.ndarray, *, seed: int, draws: int = 20_000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap requires nonempty root-level values")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(draws, values.size))
    sampled = values[index].mean(axis=1)
    return {"mean": float(values.mean()), "lower": float(np.quantile(sampled, .025)), "upper": float(np.quantile(sampled, .975))}


def _sign_flip(values: np.ndarray) -> float:
    """Exact two-sided sign-flip p value where feasible, otherwise Monte Carlo."""
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[np.abs(values) > _EPS]
    if not nonzero.size:
        return 1.0
    observed = abs(float(nonzero.mean()))
    if nonzero.size <= 20:
        signs = ((np.arange(1 << nonzero.size)[:, None] >> np.arange(nonzero.size)) & 1) * 2 - 1
        statistic = np.abs((signs * nonzero[None, :]).mean(axis=1))
        return float(np.mean(statistic >= observed - _EPS))
    rng = np.random.default_rng(202610231)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(100_000, nonzero.size))
    statistic = np.abs((signs * nonzero[None, :]).mean(axis=1))
    return float(np.mean(statistic >= observed - _EPS))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count, running, result = len(ordered), 0.0, {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        result[name] = running
    return result


def audit(decision_path: Path, outcome_path: Path) -> dict[str, Any]:
    decision = _read(decision_path)
    rows, _ = _validate_decision(decision)
    labels = _validate_outcomes(_read(outcome_path), decision_path, rows)
    groups = _root_groups(rows)
    routes = {policy: {str(root): str(arm) for root, arm in values.items()} for policy, values in decision["root_routes"].items()}
    selected = _selection(groups, labels, routes)
    roots = sorted(groups)

    summary = {}
    for policy in POLICIES:
        arm_rows = selected[policy]
        regrets = np.asarray([arm_rows[root]["regret"] for root in roots], dtype=np.float64)
        summary[policy] = {
            "root_count": len(roots), "mean_regret": float(regrets.mean()),
            "success_rate": float(np.mean([arm_rows[root]["success"] for root in roots])),
            "top1_accuracy": float(np.mean([arm_rows[root]["ranking_correct"] for root in roots])),
            "arm_selection_frequency": {arm: float(np.mean([arm_rows[root]["selected_arm"] == arm for root in roots])) for arm in ARMS},
        }
    comparisons = {}
    baseline_names = ("action_effect_only", "uniform_history_repair", "ordinary_preference", "visible_motion_control", "mse_control", "support_control", "atm_control")
    raw_p = {}
    for index, baseline in enumerate(baseline_names):
        gain = np.asarray([selected[baseline][root]["regret"] - selected["repairability_diagnostic"][root]["regret"] for root in roots])
        name = f"repairability_diagnostic_over_{baseline}"
        raw_p[name] = _sign_flip(gain)
        comparisons[name] = {
            "estimand": "root-level paired reduction in true fixed-bank regret", "gain": _bootstrap(gain, seed=202610240 + index),
            "two_sided_sign_flip_p": raw_p[name],
            "ranking_flip_gain": float(np.mean([selected[baseline][root]["ranking_correct"] is False and selected["repairability_diagnostic"][root]["ranking_correct"] is True for root in roots])),
            "success_gain": float(np.mean([selected["repairability_diagnostic"][root]["success"] for root in roots]) - np.mean([selected[baseline][root]["success"] for root in roots])),
        }
    adjusted = _holm(raw_p)
    for name, value in adjusted.items(): comparisons[name]["holm_adjusted_p"] = value
    return {
        "schema": SCHEMA,
        "claim_boundary": "Root-clustered offline fixed-bank outcome audit. It evaluates a route sealed before outcomes, not conditional safety, online MPC, contact transfer, or a universal world-model claim.",
        "integrity": {"decision_sha256": _sha256(decision_path), "outcome_sha256": _sha256(outcome_path), "outcome_bound_to_sealed_decision": True,
                      "outcome_root_count": len(roots), "outcome_labels_not_present_in_decision": True,
                      "required_arms": list(ARMS), "required_policies": list(POLICIES)},
        "methods": summary,
        "primary_estimand": "repairability_diagnostic_over_uniform_history_repair",
        "paired_comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact {args.output}")
    result = audit(args.decision, args.outcomes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "primary": result["paired_comparisons"][result["primary_estimand"]]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
