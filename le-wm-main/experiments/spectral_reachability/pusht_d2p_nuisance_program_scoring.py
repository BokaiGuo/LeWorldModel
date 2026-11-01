"""Pure saved-array candidate compiler; no model, environment, or P1 gate.

The caller supplies predictions and Jacobians evaluated at EACH prefix's own
state and suffix nominal action. This module cannot verify their provenance.
Actions are compiler coordinates, not physical commands: the CUDA caller must
apply the released packet mapping and evaluate nonlinear terminal predictions.
The finite goal-set quotient is not a guarantee of physical task fidelity.
"""
from __future__ import annotations

import hashlib

import numpy as np

from experiments.spectral_reachability.pusht_d2p_goal_nuisance_metric import compile_metric


def _array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _digest(value):
    value = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def compile_candidates(*, goal_vectors, candidate_ids, prefix_params,
                       prefix_vectors, suffix_nominal_params,
                       suffix_nominal_terminal_vectors, suffix_jacobians,
                       zero_candidate_id, damping, beta,
                       metric_by_role=None):
    """Compile unconstrained local DLS suffixes and rank stage + beta own V.

    Shapes: prefix_params [N,P,2], prefix_vectors [N,D], suffix_nominal_params
    [N,H,2], suffix_nominal_terminal_vectors [N,D], suffix_jacobians [N,D,2H].
    Jacobians differentiate terminals w.r.t. the supplied nominal parameters.
    Stage cost is the post-prefix goal cost (not accumulated trajectory cost).
    All squared errors AND ridge penalties are divided by latent dimension D.
    Optional metric_by_role must bind exactly stage/reach/terminal to the same
    goal-set projection; it is a consistency check, never an alternate metric.
    Ties retain input bank order; no non-greedy eligibility filtering occurs.
    """
    if not np.isfinite(damping) or damping <= 0 or not np.isfinite(beta) or beta < 0:
        raise ValueError("positive damping and nonnegative beta required")
    goals = _array(goal_vectors, "goal_vectors")
    projection, metric_receipt = compile_metric(goals)
    if metric_by_role is not None:
        if set(metric_by_role) != {"stage", "reach", "terminal"}:
            raise ValueError("stage/reach/terminal metric bindings required")
        for role, value in metric_by_role.items():
            supplied = _array(value, role)
            if supplied.shape != projection.shape or not np.array_equal(supplied, projection):
                raise ValueError(f"inconsistent {role} metric")
    ids = list(candidate_ids)
    if (not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids) or zero_candidate_id not in ids):
        raise ValueError("unique nonempty string candidate IDs and explicit zero control required")
    n, d = len(ids), goals.shape[1]
    prefix = _array(prefix_params, "prefix_params")
    states = _array(prefix_vectors, "prefix_vectors")
    nominal = _array(suffix_nominal_params, "suffix_nominal_params")
    terminals = _array(suffix_nominal_terminal_vectors, "suffix_nominal_terminal_vectors")
    jac = _array(suffix_jacobians, "suffix_jacobians")
    if (prefix.ndim != 3 or prefix.shape[0] != n or prefix.shape[1] < 1 or prefix.shape[2] != 2
            or nominal.ndim != 3 or nominal.shape[0] != n or nominal.shape[1] < 1 or nominal.shape[2] != 2
            or states.shape != (n, d) or terminals.shape != (n, d)
            or jac.shape != (n, d, nominal.shape[1] * 2)):
        raise ValueError("candidate/state/action/Jacobian shapes do not align")
    if np.any(prefix[ids.index(zero_candidate_id)] != 0):
        raise ValueError("explicit zero control must have zero prefix parameters")
    rows = []
    for index, candidate_id in enumerate(ids):
        stage_residual = projection @ (goals[0] - states[index])
        residual = projection @ (goals[0] - terminals[index])
        response = projection @ jac[index]
        left, singular, right = np.linalg.svd(response, full_matrices=False)
        update = right.T @ ((singular / (singular**2 + damping)) * (left.T @ residual))
        remaining = residual - response @ update
        error = float(remaining @ remaining / d)
        penalty = float(damping * (update @ update) / d)
        stage = float(stage_residual @ stage_residual / d)
        value = error + penalty
        params = nominal[index] + update.reshape(nominal.shape[1:])
        row = {"candidate_id": candidate_id, "stage_cost": stage,
               "own_reach_value": value, "linear_terminal_cost": error,
               "ridge_penalty": penalty, "objective": stage + float(beta) * value,
               "own_projected_residual": residual.tolist(),
               "own_nominal_terminal_sha256": _digest(terminals[index]),
               "own_jacobian_sha256": _digest(jac[index]),
               "singular_values": singular.tolist(), "suffix_delta_params": update.tolist(),
               "suffix_params": params.tolist(),
               "program_params": np.concatenate((prefix[index], params)).tolist()}
        if not np.isfinite([stage, value, row["objective"]]).all():
            raise ValueError("nonfinite compiled score")
        rows.append(row)
    selected = min(range(n), key=lambda i: rows[i]["objective"])
    greedy = min(range(n), key=lambda i: rows[i]["stage_cost"])
    return {"schema": "pusht_d2p_nuisance_program_scoring.v1",
            "status": "compiled_arrays_only", "P1_admission": False,
            "mechanism_pass": None, "physical_labels_used": False,
            "model_inference_executed": False, "additional_training": False,
            "own_state_provenance": "caller supplied; not verified by array scoring",
            "metric": {**metric_receipt, "projection": projection.tolist(),
                       "sha256": _digest(projection), "goal_set_sha256": _digest(goals),
                       "normalization": "squared_sum_divided_by_latent_dim",
                       "same_for_stage_reach_terminal": True},
            "damping": float(damping), "beta": float(beta),
            "objective_definition": "stage_cost + beta * own_reach_value",
            "selected_candidate_id": ids[selected], "zero_candidate_id": zero_candidate_id,
            "greedy_candidate_id": ids[greedy], "rows": rows}


def score_terminal_predictions(*, goal_vectors, terminal_vectors, projection):
    """Score caller-supplied nonlinear predictions, never physical labels.

    Returns one cost per candidate. Does not rerank the stage + beta V compiler
    objective, claim that a compiled action was evaluated, or certify fidelity.
    """
    goals = _array(goal_vectors, "goal_vectors")
    expected, _ = compile_metric(goals)
    supplied = _array(projection, "projection")
    if supplied.shape != expected.shape or not np.array_equal(supplied, expected):
        raise ValueError("inconsistent terminal metric")
    vectors = _array(terminal_vectors, "terminal_vectors")
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] != goals.shape[1]:
        raise ValueError("terminal vectors must have shape [N,D]")
    residuals = (vectors - goals[0]) @ expected.T
    costs = np.mean(residuals**2, axis=1)
    if not np.isfinite(costs).all():
        raise ValueError("nonfinite terminal score")
    return {"predicted_terminal_costs": costs.tolist(), "metric_sha256": _digest(expected),
            "normalization": "squared_sum_divided_by_latent_dim",
            "physical_labels_used": False, "P1_admission": False, "mechanism_pass": None}
