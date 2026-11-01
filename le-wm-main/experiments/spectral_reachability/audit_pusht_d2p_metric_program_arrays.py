"""Independent saved-array least-squares audit; not CUDA or physical evidence."""
from __future__ import annotations

import numpy as np


def _array(value):
    source = np.asarray(value)
    if source.dtype.kind not in "fiu" or not np.isfinite(source).all():
        raise ValueError("finite real arrays required")
    return source.astype(np.float64)


def _close(actual, expected, label, *, tolerance=2e-6):
    actual, expected = _array(actual), _array(expected)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise ValueError(f"{label} mismatch")


def audit(report):
    """Recompute ridge with augmented lstsq, never call the production scorer.

    Saved Jacobian/prediction provenance and real execution cannot be proved by
    array algebra. Nonlinear terminal costs remain diagnostics, not selection.
    """
    if (report.get("schema") != "pusht_d2p_metric_program_compiler.v1"
            or report.get("P1_admission") is not False
            or report.get("physical_labels_used") is not False
            or report.get("nonlinear_predictions_used_for_selection") is not False):
        raise ValueError("fixed-bank diagnostic contract required")
    goals = _array(report["goal_vectors"])
    if goals.ndim != 2 or goals.shape[0] < 2:
        raise ValueError("goal set required")
    differences = goals[1:] - goals[0]
    _, singular, vh = np.linalg.svd(differences, full_matrices=False)
    rank = int(np.sum(singular > max(differences.shape)*np.finfo(float).eps*singular[0])) if singular.size else 0
    projection = np.eye(goals.shape[1]) - vh[:rank].T@vh[:rank]
    scoring = report["scoring"]
    _close(scoring["metric"]["projection"], projection, "goal projection")
    damping, beta = scoring["damping"], scoring["beta"]
    if type(damping) not in (float, int) or type(beta) not in (float, int) or not np.isfinite([damping, beta]).all() or damping <= 0 or beta < 0:
        raise ValueError("positive damping and nonnegative beta required")
    rows, receipts = scoring["rows"], report["receipts"]
    if not rows or len(rows) != len(receipts):
        raise ValueError("complete candidate receipts required")
    identifiers, objectives = [], []
    for row, receipt in zip(rows, receipts, strict=True):
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in identifiers or receipt["candidate_id"] != candidate_id:
            raise ValueError("unique aligned candidate IDs required")
        identifiers.append(candidate_id)
        nominal = _array(receipt["nominal"])
        jac = _array(receipt["jacobian"])
        terminal = _array(receipt["nominal_terminal"])
        prefix_vector = _array(receipt["prefix_vector"])
        if nominal.shape != (report["suffix_horizon"], 2) or jac.shape != (goals.shape[1], nominal.size) or terminal.shape != goals[0].shape or prefix_vector.shape != goals[0].shape or np.any(nominal != 0):
            raise ValueError("own-state zero nominal/Jacobian shapes required")
        response, residual = projection@jac, projection@(goals[0]-terminal)
        augmented = np.vstack((response, np.sqrt(damping)*np.eye(nominal.size)))
        delta = np.linalg.lstsq(augmented, np.r_[residual, np.zeros(nominal.size)], rcond=None)[0]
        error = np.sum((residual-response@delta)**2)/goals.shape[1]
        penalty = damping*np.sum(delta**2)/goals.shape[1]
        stage = np.mean((projection@(goals[0]-prefix_vector))**2)
        for label, value in (("stage_cost", stage), ("linear_terminal_cost", error),
                             ("ridge_penalty", penalty), ("own_reach_value", error+penalty),
                             ("objective", stage+beta*(error+penalty))):
            _close(row[label], value, label)
        _close(row["suffix_delta_params"], delta, "delta ridge solution")
        suffix = nominal + delta.reshape(nominal.shape)
        _close(row["suffix_params"], suffix, "compiled suffix")
        _close(receipt["executed_suffix_params"], suffix, "converted suffix")
        program = np.concatenate((_array(receipt["prefix_params"]), suffix))
        _close(receipt["executed_program_params"], program, "program parameters")
        _close(row["program_params"], program, "scoring program")
        _close(receipt["program_live_commands"], np.tanh(program), "live commands")
        predicted = _array(receipt["predicted_terminal_vector"])
        diagnostic = np.mean((projection@(predicted-goals[0]))**2)
        _close(receipt["nonlinear_terminal_cost_diagnostic_only"], diagnostic, "nonlinear diagnostic")
        objectives.append(stage+beta*(error+penalty))
    selected = identifiers[int(np.argmin(objectives))]
    if selected != scoring["selected_candidate_id"] or selected != report["selected_candidate_id"]:
        raise ValueError("stage plus own-V selection mismatch")
    return {"status": "saved_array_algebra_verified", "candidates": len(rows),
            "independent_augmented_lstsq_fits": len(rows), "selected_candidate_id": selected,
            "P1_admission": False, "scope": "array consistency only; not model fidelity or physical benefit"}
