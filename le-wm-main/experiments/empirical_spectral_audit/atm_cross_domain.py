"""ATM-style cross-domain action diagnostics for LeWorldModel.

The implementation intentionally separates three questions:

1. Can a probe recover actions from the current state alone (policy shortcut)?
2. Can it recover actions from transition effects in the true/predicted domain?
3. Do predicted task-cost differences preserve real action ordering?

The four inverse risks are descriptive diagnostics.  They are not a causal
certificate and do not authorize planning without independent intervention and
closed-loop evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def transition_features(source: np.ndarray, successor: np.ndarray, *, effect_only: bool = False) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    successor = np.asarray(successor, dtype=np.float64)
    if source.ndim != 2 or successor.shape != source.shape:
        raise ValueError("source and successor must be equal-shaped rank-2 arrays")
    delta = successor - source
    if effect_only:
        return delta
    return np.concatenate((source, successor, delta), axis=1)


def _fit_ridge(features: np.ndarray, actions: np.ndarray, alpha: float) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    gram = design.T @ design
    gram += float(alpha) * np.eye(gram.shape[0])
    gram[-1, -1] -= float(alpha)  # do not regularize the intercept
    return np.linalg.solve(gram, design.T @ actions)


def _risk(features: np.ndarray, actions: np.ndarray, weights: np.ndarray) -> float:
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    prediction = design @ weights
    return float(np.mean((prediction - np.asarray(actions, dtype=np.float64)) ** 2))


def action_information_gain(prior_risk: float, successor_risk: float) -> dict[str, Any]:
    """Report successor information beyond a state/history-only action prior.

    Under squared loss this is a risk reduction, not Shannon mutual
    information.  It is only interpretable under the declared intervention
    distribution and does not establish causal task-effect correctness.
    """

    prior = float(prior_risk)
    successor = float(successor_risk)
    if not np.isfinite(prior) or not np.isfinite(successor) or prior < 0.0 or successor < 0.0:
        raise ValueError("risks must be finite and non-negative")
    return {
        "schema": "interventional_action_information_gain.v1",
        "prior_risk": prior,
        "successor_risk": successor,
        "delta_action_information": prior - successor,
        "semantics": "squared-loss Bayes-risk reduction, not Shannon mutual information",
        "claim_boundary": "requires balanced intervention support; does not certify task-effect transfer",
    }


def inverse_risk_matrix(
    source: np.ndarray,
    true_successor: np.ndarray,
    predicted_successor: np.ndarray,
    actions: np.ndarray,
    *,
    fit_mask: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    alpha: float = 1.0e-6,
) -> dict[str, Any]:
    """Fit true/predicted-domain probes and cross-evaluate them.

    Matrix convention is rows = probe training domain and columns = test
    domain: ``[[D_TT, D_TP], [D_PT, D_PP]]``.
    """

    source = np.asarray(source, dtype=np.float64)
    true_successor = np.asarray(true_successor, dtype=np.float64)
    predicted_successor = np.asarray(predicted_successor, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    n = len(source)
    if any(array.ndim != 2 or len(array) != n for array in (source, true_successor, predicted_successor, actions)):
        raise ValueError("all inputs must be aligned rank-2 arrays")
    fit = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    evaluate = np.ones(n, dtype=bool) if eval_mask is None else np.asarray(eval_mask, dtype=bool)
    if fit.shape != (n,) or evaluate.shape != (n,) or not np.any(fit) or not np.any(evaluate):
        raise ValueError("fit/eval masks must be non-empty vectors aligned with inputs")

    true = transition_features(source, true_successor)
    predicted = transition_features(source, predicted_successor)
    true_probe = _fit_ridge(true[fit], actions[fit], alpha)
    predicted_probe = _fit_ridge(predicted[fit], actions[fit], alpha)
    matrix = np.asarray(
        [
            [_risk(true[evaluate], actions[evaluate], true_probe), _risk(predicted[evaluate], actions[evaluate], true_probe)],
            [_risk(true[evaluate], actions[evaluate], predicted_probe), _risk(predicted[evaluate], actions[evaluate], predicted_probe)],
        ],
        dtype=np.float64,
    )
    prior_probe = _fit_ridge(source[fit], actions[fit], alpha)
    effect_true_probe = _fit_ridge(transition_features(source, true_successor, effect_only=True)[fit], actions[fit], alpha)
    effect_predicted_probe = _fit_ridge(transition_features(source, predicted_successor, effect_only=True)[fit], actions[fit], alpha)
    prior_risk = _risk(source[evaluate], actions[evaluate], prior_probe)
    effect_matrix = np.asarray(
        [
            [_risk((true_successor - source)[evaluate], actions[evaluate], effect_true_probe), _risk((predicted_successor - source)[evaluate], actions[evaluate], effect_true_probe)],
            [_risk((true_successor - source)[evaluate], actions[evaluate], effect_predicted_probe), _risk((predicted_successor - source)[evaluate], actions[evaluate], effect_predicted_probe)],
        ],
        dtype=np.float64,
    )
    return {
        "schema": "atm_cross_domain_diagnostic.v1",
        "inverse_risk_matrix": matrix.tolist(),
        "effect_only_inverse_risk_matrix": effect_matrix.tolist(),
        "state_only_action_prior_risk": prior_risk,
        "action_information_gain": action_information_gain(
            prior_risk,
            float(matrix[0, 0]),
        ),
        "cross_domain_gaps": {
            "true_probe_transfer_gap": float(matrix[0, 1] - matrix[0, 0]),
            "predicted_probe_transfer_gap": float(matrix[1, 0] - matrix[1, 1]),
            "diagonal_mean": float(np.mean(np.diag(matrix))),
            "off_diagonal_mean": float(np.mean((matrix[0, 1], matrix[1, 0]))),
        },
        "protocol": {
            "probe_rows": "true,predicted",
            "test_columns": "true,predicted",
            "fit_outcome_disjoint": True,
            "state_only_shortcut_reported": True,
            "effect_only_intervention_probe_reported": True,
            "claim_boundary": "inverse consistency is diagnostic; it is not task-effect or planning certification",
        },
    }


def action_ranking_inversion(
    real_cost: np.ndarray,
    predicted_cost: np.ndarray,
    *,
    group_id: np.ndarray,
    minimum_margin: float = 0.0,
) -> dict[str, Any]:
    """Measure finite-horizon candidate ranking failures, grouped by root."""

    real_cost = np.asarray(real_cost, dtype=np.float64).reshape(-1)
    predicted_cost = np.asarray(predicted_cost, dtype=np.float64).reshape(-1)
    group_id = np.asarray(group_id).reshape(-1)
    if not (len(real_cost) == len(predicted_cost) == len(group_id)):
        raise ValueError("cost and group arrays must have equal length")
    inversions = 0
    comparable = 0
    regrets = []
    for group in np.unique(group_id):
        indices = np.flatnonzero(group_id == group)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                i, j = indices[left], indices[right]
                real_diff = float(real_cost[i] - real_cost[j])
                pred_diff = float(predicted_cost[i] - predicted_cost[j])
                if abs(real_diff) <= float(minimum_margin):
                    continue
                comparable += 1
                inversions += int(np.sign(real_diff) != np.sign(pred_diff))
        best_real = float(np.min(real_cost[indices]))
        selected = int(indices[np.argmin(predicted_cost[indices])])
        regrets.append(float(real_cost[selected] - best_real))
    return {
        "schema": "finite_horizon_action_ranking.v1",
        "comparable_pair_count": comparable,
        "inversion_count": inversions,
        "inversion_rate": None if comparable == 0 else float(inversions / comparable),
        "mean_selected_regret": float(np.mean(regrets)) if regrets else None,
        "minimum_decision_margin": float(minimum_margin),
        "claim_boundary": "candidate ranking is evaluated on independent real task costs; it is not inferred from inverse risk alone",
    }
