"""Execution-grounded v2 repairability frontier for privileged PointMaze controls.

Unlike v1, the router sees cross-fitted train-root repair labels: each label
comes from selectors trained without that root.  This tests whether v1 failed
because the router had only 20 GDD labels, noisy in-sample labels, weak lawful
features, or a one-sided MLP-only release decision.  It is a predeclared
control protocol, not a teacher or deployment claim.

The outcome split is deliberately not executed until after all feature family,
regularization, selector, and calibration-threshold choices have frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.gdd_wm import run_pointmaze_repairability_frontier_pilot as v1
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_execution_grounded_repairability_frontier.v2"
CONTROL_LABEL = "privileged_execution_supervised_state_action_control_v2"
FEATURE_FAMILIES = {
    "margin_disagreement": (0, 1, 2, 3, 4, 5, 6),
    "action_support": (7, 8, 9, 10),
    "state_goal": (11, 12, 13),
    "all_lawful_features": tuple(range(14)),
}
L2_GRID = (0.01, 0.1, 1.0, 10.0)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _root_rows(groups: dict[str, list[dict[str, Any]]], roots: list[str]) -> list[dict[str, Any]]:
    return [row for root in roots for row in groups[root]]


def _cost_map(rows: list[dict[str, Any]], costs: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = defaultdict(list)
    for row, cost in zip(rows, costs):
        result[str(row["instance_id"])].append(float(cost))
    return {root: np.asarray(value, dtype=np.float64) for root, value in result.items()}


def _fit_selectors(rows: list[dict[str, Any]], costs: np.ndarray, terminal: np.ndarray, *, width: int, device: str, seed: int) -> tuple[Callable[[list[dict[str, Any]]], dict[str, np.ndarray]], dict[str, Any]]:
    """Fit PSD, ordinary cost MLP, and naive preference control on named roots."""

    psd = v1._fit_ridge(np.stack([v1._state_action_features(row) for row in rows]), terminal)
    features = np.stack([v1._features(row) for row in rows])
    _model, mlp = v1._torch_mlp_fit(features, costs, width=width, steps=300, device=device, seed=seed)
    naive = v1._fit_pairwise(features, costs, [str(row["instance_id"]) for row in rows], device=device, seed=seed + 17)

    def score(group: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        state_action = np.stack([v1._state_action_features(row) for row in group])
        terminal_prediction = v1._ridge_predict(state_action, psd)
        goal = np.stack([np.asarray(row["goal_raw"], dtype=np.float64) for row in group])
        features_group = np.stack([v1._features(row) for row in group])
        return {
            "psd": -np.linalg.norm(terminal_prediction[:, :2] - goal, axis=1),
            "mlp": -mlp(features_group),
            "naive": naive(features_group),
        }

    return score, {"mlp_width": width, "fit_row_count": len(rows), "feature_dim": int(features.shape[1])}


def _top_gap(scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    return float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0


def _support_features(group: list[dict[str, Any]], train_actions: np.ndarray) -> tuple[float, float, float, float]:
    actions = np.stack([v1._action_summary(row) for row in group])
    # Training roots define the only support reference.  It contains no held
    # action outcome, candidate identity, or task cost.
    distances = np.linalg.norm(actions[:, None, :] - train_actions[None, :, :], axis=2).min(axis=1)
    pairwise = np.linalg.norm(actions[:, None, :] - actions[None, :, :], axis=2)
    upper = pairwise[np.triu_indices(len(actions), k=1)]
    return float(distances.min()), float(distances.mean()), float(distances.max()), float(upper.mean()) if len(upper) else 0.0


def _router_features(group: list[dict[str, Any]], scores: dict[str, np.ndarray], train_actions: np.ndarray) -> np.ndarray:
    psd, mlp = scores["psd"], scores["mlp"]
    psd_pick, mlp_pick = v1._choose(group, psd), v1._choose(group, mlp)
    state = np.asarray(group[0]["source_state_raw"], dtype=np.float64)
    goal = np.asarray(group[0]["goal_raw"], dtype=np.float64)
    support = _support_features(group, train_actions)
    return np.asarray([
        float(psd[psd_pick] - psd[mlp_pick]),
        float(mlp[mlp_pick] - mlp[psd_pick]),
        float(psd_pick != mlp_pick),
        _top_gap(psd),
        _top_gap(mlp),
        float(np.std(psd)),
        float(np.std(mlp)),
        *support,
        float(np.linalg.norm(state[2:])),
        float(np.linalg.norm(state[:2] - goal)),
        float(np.linalg.norm(state[:2])),
    ], dtype=np.float64)


def _fit_logistic(features: np.ndarray, labels: np.ndarray, *, l2: float) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Robustly standardized, L2-regularized deterministic logistic router."""

    if l2 <= 0.0:
        raise ValueError("l2 must be positive")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    if len(np.unique(labels)) < 2:
        probability = float(np.mean(labels))
        return (lambda values: np.full(len(values), probability, dtype=np.float64)), {
            "constant_probability": probability, "l2": l2, "feature_mean": mean, "feature_scale": scale,
        }
    design = np.concatenate((np.ones((len(standardized), 1)), standardized), axis=1)
    beta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(800):
        logits = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - labels) / len(labels) + l2 * np.r_[0.0, beta[1:]] / len(labels)
        hessian = design.T @ (design * (probability * (1.0 - probability))[:, None]) / len(labels)
        hessian += np.diag(np.r_[0.0, np.full(design.shape[1] - 1, l2 / len(labels))])
        try:
            step = np.linalg.solve(hessian + np.eye(len(beta)) * 1.0e-8, gradient)
        except np.linalg.LinAlgError:
            step = gradient
        beta -= np.clip(step, -1.0, 1.0)
        if float(np.max(np.abs(step))) < 1.0e-8:
            break

    def predict(values: np.ndarray) -> np.ndarray:
        x = (values - mean) / scale
        design_eval = np.concatenate((np.ones((len(x), 1)), x), axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(design_eval @ beta, -30.0, 30.0)))

    return predict, {"l2": l2, "feature_mean": mean, "feature_scale": scale, "beta": beta}


def _auc(probability: np.ndarray, labels: np.ndarray) -> float | None:
    positive, negative = probability[labels > 0.5], probability[labels <= 0.5]
    if not len(positive) or not len(negative):
        return None
    comparisons = (positive[:, None] > negative[None, :]).astype(float) + 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean())


def _crossfit_router_labels(groups: dict[str, list[dict[str, Any]]], costs: dict[str, np.ndarray], terminal: dict[str, np.ndarray], train_roots: list[str], *, width: int, device: str, seed: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float], dict[str, Any]]:
    """Produce out-of-fold repair labels and lawful router features for all train roots."""

    folds: dict[str, int] = {root: int(_hash(root)[:8], 16) % 4 for root in train_roots}
    actions_all = np.stack([v1._action_summary(row) for root in train_roots for row in groups[root]])
    scores_out: dict[str, dict[str, np.ndarray]] = {}
    label_out: dict[str, float] = {}
    feature_out: dict[str, np.ndarray] = {}
    for fold in range(4):
        held, fit = [root for root in train_roots if folds[root] == fold], [root for root in train_roots if folds[root] != fold]
        if not held or not fit:
            continue
        fit_rows = _root_rows(groups, fit)
        fit_cost = np.concatenate([costs[root] for root in fit])
        fit_terminal = np.concatenate([terminal[root] for root in fit])
        scorer, _ = _fit_selectors(fit_rows, fit_cost, fit_terminal, width=width, device=device, seed=seed + fold)
        fit_actions = np.stack([v1._action_summary(row) for row in fit_rows])
        for root in held:
            score = scorer(groups[root])
            psd_pick, mlp_pick = v1._choose(groups[root], score["psd"]), v1._choose(groups[root], score["mlp"])
            scores_out[root] = score
            label_out[root] = float(costs[root][mlp_pick] + 1.0e-10 < costs[root][psd_pick])
            feature_out[root] = _router_features(groups[root], score, fit_actions)
    if set(scores_out) != set(train_roots):
        raise ValueError("cross-fit did not emit each train root exactly once")
    return scores_out, label_out, {"features": feature_out, "folds": folds, "full_train_action_count": len(actions_all)}


def _router_decision(groups: dict[str, list[dict[str, Any]]], scores: dict[str, dict[str, np.ndarray]], costs: dict[str, np.ndarray], probabilities: dict[str, float], threshold: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for root, group in groups.items():
        psd_pick, mlp_pick = v1._choose(group, scores[root]["psd"]), v1._choose(group, scores[root]["mlp"])
        naive_pick = v1._choose(group, scores[root]["naive"])
        # Calibration chooses the release threshold.  Therefore the router can
        # choose the lower calibrated-risk candidate (MLP) or retain PSD.
        routed_pick = mlp_pick if probabilities[root] >= threshold else psd_pick
        oracle = int(np.argmin(costs[root]))
        rows[root] = {
            "router_probability_pre_outcome": float(probabilities[root]),
            "router_selected_mlp_pre_outcome": bool(probabilities[root] >= threshold),
            "psd_regret": float(costs[root][psd_pick] - costs[root][oracle]),
            "mlp_regret": float(costs[root][mlp_pick] - costs[root][oracle]),
            "naive_regret": float(costs[root][naive_pick] - costs[root][oracle]),
            "routed_regret": float(costs[root][routed_pick] - costs[root][oracle]),
            "repair_outcome_mlp_beats_psd": bool(costs[root][mlp_pick] + 1.0e-10 < costs[root][psd_pick]),
        }
    return {"rows": rows, "outcome_cost_used_for_selection": False}


def _summarize(decision: dict[str, Any]) -> dict[str, Any]:
    rows = list(decision["rows"].values())
    repair = np.asarray([row["repair_outcome_mlp_beats_psd"] for row in rows], dtype=bool)
    routed = np.asarray([row["router_selected_mlp_pre_outcome"] for row in rows], dtype=bool)
    regret = {method: float(np.mean([row[f"{method}_regret"] for row in rows])) for method in ("psd", "mlp", "naive", "routed")}
    precision = None if not routed.any() else float(repair[routed].mean())
    prevalence = float(repair.mean())
    return {
        "root_count": len(rows),
        "mean_regret": regret,
        "router_coverage": float(routed.mean()),
        "repair_outcome_prevalence": prevalence,
        "preselection_repair_precision": precision,
        "preselection_repair_recall": None if not repair.any() else float(routed[repair].mean()),
        "preselection_prediction_verdict": {
            "status": "confirmed_in_this_pilot" if precision is not None and precision > prevalence + 0.05 and regret["routed"] < regret["psd"] - 1.0e-10 else "not_confirmed_in_this_pilot",
            "criterion": "precision exceeds repair prevalence by >0.05 and routed mean regret improves over PSD",
        },
        "outcome_cost_used_for_selection": False,
    }


def _select_calibration_threshold(
    groups: dict[str, list[dict[str, Any]]],
    scores: dict[str, dict[str, np.ndarray]],
    costs: dict[str, np.ndarray],
    probabilities: dict[str, float],
) -> tuple[float, float, dict[str, Any]]:
    """Freeze the best calibration threshold and its matching decision.

    Keeping the selected decision with the threshold prevents the common bug
    where a later grid iteration is accidentally reported as the calibrated
    policy.
    """

    frontier = _calibration_threshold_frontier(groups, scores, costs, probabilities)
    best_threshold, best_regret, selected_decision = 1.0, float("inf"), None
    for candidate in frontier:
        threshold = float(candidate["threshold"])
        decision = candidate["decision"]
        value = float(candidate["mean_routed_regret"])
        if value < best_regret - 1.0e-12 or (np.isclose(value, best_regret) and threshold > best_threshold):
            best_threshold, best_regret, selected_decision = float(threshold), float(value), decision
    if selected_decision is None:
        raise ValueError("calibration threshold selection received no candidates")
    return best_threshold, best_regret, selected_decision


def _calibration_threshold_frontier(
    groups: dict[str, list[dict[str, Any]]],
    scores: dict[str, dict[str, np.ndarray]],
    costs: dict[str, np.ndarray],
    probabilities: dict[str, float],
) -> list[dict[str, Any]]:
    """Materialize the calibration-only risk/coverage frontier.

    A zero-coverage selected route can be the correct fail-closed decision, but
    it is scientifically ambiguous without knowing whether the MLP ever
    disagreed with PSD, whether any nonempty release was beneficial, or whether
    the probability score simply failed to isolate those roots.  This helper
    records those distinctions without reading outcome roots.
    """

    thresholds = sorted({0.0, 0.25, 0.5, 0.75, 1.0, *probabilities.values()})
    frontier: list[dict[str, Any]] = []
    for threshold in thresholds:
        decision = _router_decision(groups, scores, costs, probabilities, threshold)
        summary = _summarize(decision)
        rows = list(decision["rows"].values())
        selected = [row for row in rows if bool(row["router_selected_mlp_pre_outcome"])]
        disagreement_rate = float(np.mean([
            not np.isclose(row["mlp_regret"], row["psd_regret"])
            for row in rows
        ]))
        frontier.append({
            "threshold": float(threshold),
            "decision": decision,
            "mean_routed_regret": float(summary["mean_regret"]["routed"]),
            "routed_minus_psd_regret": float(
                summary["mean_regret"]["routed"] - summary["mean_regret"]["psd"]
            ),
            "router_coverage": float(summary["router_coverage"]),
            "selected_root_count": int(len(selected)),
            "mlp_psd_regret_disagreement_rate": disagreement_rate,
            "selected_repair_prevalence": (
                None if not selected else float(np.mean([
                    row["repair_outcome_mlp_beats_psd"] for row in selected
                ]))
            ),
        })
    return frontier


def _public_calibration_frontier(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip executable decisions while retaining auditable calibration data."""

    return [{key: value for key, value in candidate.items() if key != "decision"} for candidate in frontier]


def _diagnose(crossfit_labels: dict[str, float], full_train_labels: dict[str, float], ablation: dict[str, Any], selected: dict[str, Any], calibration: dict[str, Any], *, gdd_root_count: int, calibration_frontier: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    agreement = float(np.mean([crossfit_labels[root] == full_train_labels[root] for root in crossfit_labels]))
    selected_auc = selected.get("gdd_auc")
    causes: list[str] = []
    if agreement < 0.7:
        causes.append("cross_fitted_repair_labels_are_unstable")
    if selected_auc is None or selected_auc <= 0.55:
        causes.append("lawful_router_features_do_not_discriminate_repair_on_gdd")
    if float(np.mean(list(crossfit_labels.values()))) < 0.15:
        causes.append("repair_events_are_rare_in_cross_fitted_train_roots")
    if (ablation.get("action_support", {}).get("gdd_auc") or 0.0) <= 0.55:
        causes.append("action_bank_support_alone_has_no_GDD_repair_discrimination")
    if float(calibration.get("router_coverage", 0.0)) == 0.0:
        causes.append("calibration_supports_abstention_only_not_MLP_release")
        nonempty = [candidate for candidate in (calibration_frontier or []) if candidate["selected_root_count"] > 0]
        if nonempty and all(candidate["mlp_psd_regret_disagreement_rate"] == 0.0 for candidate in nonempty):
            causes.append("MLP_and_PSD_never_change_calibration_regret")
        elif nonempty and all(candidate["routed_minus_psd_regret"] >= -1.0e-12 for candidate in nonempty):
            causes.append("no_nonempty_calibration_release_improves_over_PSD")
    if not causes:
        causes.append("no_single_pre_outcome_failure_signal_dominates; require independent replication")
    return {
        "crossfit_train_root_count": len(crossfit_labels),
        "gdd_root_count": gdd_root_count,
        "crossfit_label_prevalence": float(np.mean(list(crossfit_labels.values()))),
        "crossfit_vs_full_train_label_agreement": agreement,
        "gdd_feature_ablation": ablation,
        "diagnostic_signals": causes,
        "interpretation": "These are pre-outcome diagnostic signals only. They distinguish label instability, feature-family failure, rare repair events, and calibrated abstention without selecting a held-out outcome configuration.",
    }


def run(candidate_dataset: Path, output: Path, *, device: str = "cuda", seed: int = 8412) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata, rows = v1._load_records(candidate_dataset)
    roles = v1._assign_roles(rows)
    groups = v1._groups(rows, roles)
    cuda = require_cuda_allocation(device)
    root_by_role = {role: sorted(root for root, assigned in roles.items() if assigned == role) for role in v1.ROLES}
    train_roots, gdd_roots, calibration_roots, outcome_roots = (root_by_role[role] for role in v1.ROLES)

    # Fresh execution begins with non-outcome roots only.  The program has no
    # outcome cost array until after choice of feature family/L2/threshold.
    nonoutcome_roots = train_roots + gdd_roots + calibration_roots
    nonoutcome_rows = _root_rows(groups, nonoutcome_roots)
    nonoutcome_terminal = v1._fresh_execution(nonoutcome_rows)
    nonoutcome_cost = v1._terminal_cost(nonoutcome_rows, nonoutcome_terminal)
    costs = _cost_map(nonoutcome_rows, nonoutcome_cost)
    terminal_by_root: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, terminal in zip(nonoutcome_rows, nonoutcome_terminal):
        terminal_by_root[str(row["instance_id"])].append(np.asarray(terminal, dtype=np.float64))
    terminal = {root: np.stack(values) for root, values in terminal_by_root.items()}

    train_rows = _root_rows(groups, train_roots)
    train_cost, train_terminal = np.concatenate([costs[root] for root in train_roots]), np.concatenate([terminal[root] for root in train_roots])
    gdd_rows, gdd_cost = _root_rows(groups, gdd_roots), np.concatenate([costs[root] for root in gdd_roots])
    width_options = (8, 16)
    width_candidates: list[dict[str, Any]] = []
    for width in width_options:
        scorer, details = _fit_selectors(train_rows, train_cost, train_terminal, width=width, device=device, seed=seed + width)
        prediction = np.concatenate([-scorer(groups[root])["mlp"] for root in gdd_roots])
        width_candidates.append({"width": width, "gdd_cost_mse": float(np.mean((prediction - gdd_cost) ** 2)), "scorer": scorer, "details": details})
    chosen = min(width_candidates, key=lambda item: (item["gdd_cost_mse"], item["width"]))
    final_scorer = chosen["scorer"]
    train_actions = np.stack([v1._action_summary(row) for row in train_rows])

    cross_scores, cross_labels, cross_detail = _crossfit_router_labels(groups, costs, terminal, train_roots, width=int(chosen["width"]), device=device, seed=seed + 100)
    cross_features = {root: np.asarray(cross_detail["features"][root], dtype=np.float64) for root in train_roots}
    full_scores = {root: final_scorer(groups[root]) for root in nonoutcome_roots}
    full_features = {root: _router_features(groups[root], full_scores[root], train_actions) for root in nonoutcome_roots}
    full_train_labels: dict[str, float] = {}
    for root in train_roots:
        psd, mlp = v1._choose(groups[root], full_scores[root]["psd"]), v1._choose(groups[root], full_scores[root]["mlp"])
        full_train_labels[root] = float(costs[root][mlp] + 1.0e-10 < costs[root][psd])
    gdd_labels = []
    for root in gdd_roots:
        psd, mlp = v1._choose(groups[root], full_scores[root]["psd"]), v1._choose(groups[root], full_scores[root]["mlp"])
        gdd_labels.append(float(costs[root][mlp] + 1.0e-10 < costs[root][psd]))
    gdd_labels_array = np.asarray(gdd_labels, dtype=np.float64)

    ablation: dict[str, Any] = {}
    fitted: list[dict[str, Any]] = []
    for family, indices in FEATURE_FAMILIES.items():
        x_train = np.stack([cross_features[root][list(indices)] for root in train_roots])
        x_gdd = np.stack([full_features[root][list(indices)] for root in gdd_roots])
        variants = []
        for l2 in L2_GRID:
            predictor, detail = _fit_logistic(x_train, np.asarray([cross_labels[root] for root in train_roots]), l2=l2)
            probability = predictor(x_gdd)
            variants.append({"l2": l2, "auc": _auc(probability, gdd_labels_array), "brier": float(np.mean((probability - gdd_labels_array) ** 2)), "predictor": predictor, "detail": detail})
        best = min(variants, key=lambda item: (float("inf") if item["auc"] is None else -item["auc"], item["brier"], -item["l2"]))
        ablation[family] = {"feature_count": len(indices), "best_l2": best["l2"], "gdd_auc": best["auc"], "gdd_brier": best["brier"]}
        fitted.append({"family": family, "indices": indices, **best})
    # GDD exclusively selects the router family and L2.  Ties choose fewer
    # features, then stronger regularization, to prevent free complexity.
    selected = min(fitted, key=lambda item: (float("inf") if item["auc"] is None else -item["auc"], item["brier"], len(item["indices"]), -item["l2"]))
    router_predictor = selected["predictor"]
    probabilities = {root: float(router_predictor(full_features[root][list(selected["indices"])].reshape(1, -1))[0]) for root in nonoutcome_roots}
    calibration_frontier = _calibration_threshold_frontier(
        {root: groups[root] for root in calibration_roots},
        full_scores,
        costs,
        probabilities,
    )
    best_threshold, best_regret, calibration_decision = _select_calibration_threshold(
        {root: groups[root] for root in calibration_roots},
        full_scores,
        costs,
        probabilities,
    )

    # Freeze completed: only now execute untouched outcome roots and score the
    # fixed candidate selections.  No outcome result feeds any configuration.
    outcome_rows = _root_rows(groups, outcome_roots)
    outcome_terminal = v1._fresh_execution(outcome_rows)
    outcome_cost = v1._terminal_cost(outcome_rows, outcome_terminal)
    outcome_costs = _cost_map(outcome_rows, outcome_cost)
    outcome_scores = {root: final_scorer(groups[root]) for root in outcome_roots}
    outcome_features = {root: _router_features(groups[root], outcome_scores[root], train_actions) for root in outcome_roots}
    outcome_probability = {root: float(router_predictor(outcome_features[root][list(selected["indices"])].reshape(1, -1))[0]) for root in outcome_roots}
    outcome = _router_decision({root: groups[root] for root in outcome_roots}, outcome_scores, outcome_costs, outcome_probability, best_threshold)
    calibration_summary = _summarize(calibration_decision)
    report = {
        "schema": SCHEMA,
        "status": "completed_execution_grounded_privileged_control_v2",
        "claim_boundary": "A privileged simulator-state, execution-supervised PointMaze repairability control. It does not validate a GDD teacher, frozen visual interface, world-model prediction benefit, distillation benefit, deployment, or universal repairability.",
        "control_contract": {
            "label": CONTROL_LABEL,
            "privileged_inputs": ["source_state_raw", "goal_raw"],
            "execution_supervision": "fresh legacy PointMaze MuJoCo terminal state and goal-distance cost",
            "outcome_execution_deferred_until_frozen": True,
            "source_dataset_real_task_cost_used": False,
            "prediction_preservation": {"applicable": False, "reason": "No frozen encoder or predictor is modified; this is selector/control diagnosis."},
        },
        "split_isolation": {"roles": root_by_role, "root_disjoint": True, "train": "base selectors and cross-fitted repair labels", "gdd": "MLP width and router family/L2 selection", "calibration": "threshold only", "outcome": "fresh execution only after freeze"},
        "preselection_contract": {"router": "choose MLP only when frozen repair probability reaches calibration-selected threshold; otherwise retain PSD", "threshold": best_threshold, "outcome_cost_used_for_selection": False},
        "selector_configuration": {"width_candidates_gdd_mse": [{"width": item["width"], "gdd_cost_mse": item["gdd_cost_mse"]} for item in width_candidates], "selected_mlp_width": chosen["width"], "selected_router_family": selected["family"], "selected_router_l2": selected["l2"], "selected_router_gdd_auc": selected["auc"]},
        "diagnosis": _diagnose(cross_labels, full_train_labels, ablation, {"gdd_auc": selected["auc"]}, calibration_summary, gdd_root_count=len(gdd_roots), calibration_frontier=calibration_frontier),
        "calibration": {"threshold": best_threshold, "mean_routed_regret": best_regret, "summary": calibration_summary, "threshold_frontier": _public_calibration_frontier(calibration_frontier)},
        "outcome": _summarize(outcome),
        "outcome_per_root": outcome["rows"],
        "cuda_preflight": cuda,
        "provenance": {"candidate_dataset": str(candidate_dataset.resolve()), "candidate_dataset_sha256": v1._file_hash(candidate_dataset), "world_model_sha256": metadata.get("world_model_sha256")},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=8412)
    args = parser.parse_args()
    report = run(args.candidate_dataset, args.output, device=args.device, seed=args.seed)
    print(json.dumps({"status": report["status"], "diagnosis": report["diagnosis"], "outcome": report["outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
