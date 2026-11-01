"""Pure NumPy isometric PushT packet coordinates; no model/runtime imports.

Constant coordinates u map to ten logits z=C u, C=tile(I2,(5,1))/sqrt(5).
Full coordinates use I10. Commands are tanh(z).reshape(5,2), time-major.
The budget is ONE Euclidean norm over all logits in the supplied suffix, not
ten independent component limits or a separate radius for each horizon.
"""
from __future__ import annotations

import numpy as np


def _array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite real numeric data, not bool/complex/object')
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} exceeds float64 range')
    return array


def _dimension(chart):
    if chart not in ('constant', 'full'):
        raise ValueError("chart must be 'constant' or 'full'")
    return 2 if chart == 'constant' else 10


def _params(params, chart):
    array = _array(params, 'params')
    if array.ndim < 1 or array.shape[-1] != _dimension(chart) or array.size == 0:
        raise ValueError('params require nonempty trailing chart dimension (2 or 10)')
    return array


def _nonnegative(value, name):
    array = _array(value, name)
    if array.ndim != 0 or array < 0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return float(array)


def _norm(array):
    # Scaled reduction avoids squaring overflow for otherwise usable inputs.
    scale = np.max(np.abs(array))
    if scale == 0:
        return 0.0
    with np.errstate(over='ignore'):
        value = float(scale * np.sqrt(np.sum((array / scale) ** 2)))
    if not np.isfinite(value):
        raise ValueError('total Euclidean norm exceeds float64 range')
    return value


def chart_matrix(chart, *, horizon=1):
    """Return fresh orthonormal columns, shape (10*H, d*H), packet-major.

    horizon is a positive Python int; C_H = kron(I_H, C), F_H = I_(10H).
    """
    width = _dimension(chart)
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer, not bool')
    base = np.tile(np.eye(2), (5, 1)) / np.sqrt(5) if width == 2 else np.eye(10)
    return np.kron(np.eye(horizon), base)


def basis(kind):
    """Single-packet public entry: (10,2) constant or (10,10) full basis."""
    return chart_matrix(kind)


def params_to_full(params, chart):
    """(...,d) -> (...,10) TANH-PREIMAGE logits, not raw/model actions."""
    return _params(params, chart) @ chart_matrix(chart).T


def constant_to_full(params):
    """Isometrically embed constant coefficients; never repeat them unscaled."""
    return params_to_full(params, 'constant')


def full_to_params(logits, chart, *, atol=1e-12):
    """Strict inverse on the chart subspace, NOT a silent least-squares projection."""
    logits = _params(logits, 'full')
    atol = _nonnegative(atol, 'atol')
    matrix = chart_matrix(chart)
    coefficients = logits @ matrix
    if not np.allclose(coefficients @ matrix.T, logits, rtol=0, atol=atol):
        raise ValueError('logits are outside the requested chart subspace')
    return coefficients


def atanh_l2_norm(params, chart):
    """Total ||z||_2 across every packet; equals ||params||_2 by isometry.

    Computed before tanh, avoiding loss of information at floating saturation.
    Leading dimensions are all charged together, NOT treated as batch budgets.
    """
    return _norm(_params(params, chart))


def project_params(params, chart, *, radius):
    """Radially project onto one total atanh-logit L2 ball, without input mutation."""
    array = _params(params, chart)
    radius = _nonnegative(radius, 'radius')
    norm = _norm(array)
    if norm > radius:
        array *= radius / norm
        # Avoid a roundoff-only rejection when this result is passed straight
        # to params_to_raw_packets(..., radius=the_same_radius).
        while _norm(array) > radius:
            array *= np.nextafter(1., 0.)
    return array


def params_to_raw_packets(params, chart, *, radius=None):
    """(...,d) -> (...,5,2) raw relative commands; no action normalization.

    Optional radius rejects an over-budget suffix, never silently projects.
    Float64 tanh saturation is rejected so a strict atanh inverse remains valid.
    """
    coefficients = _params(params, chart)
    if radius is not None and _norm(coefficients) > _nonnegative(radius, 'radius'):
        raise ValueError('total atanh-logit L2 budget exceeded')
    raw = np.tanh(params_to_full(coefficients, chart))
    if np.any(np.abs(raw) >= 1):
        raise ValueError('tanh saturated at float64 endpoint; reduce logit norm')
    return raw.reshape(*coefficients.shape[:-1], 5, 2)


def raw_packets_to_params(raw_packets, chart, *, atol=1e-12):
    """Strict atanh/chart inverse; endpoints, clipping and nonconstant projection forbidden."""
    raw = _array(raw_packets, 'raw_packets')
    if raw.ndim < 2 or raw.shape[-2:] != (5, 2) or raw.size == 0:
        raise ValueError('raw_packets require nonempty trailing shape (5,2)')
    if np.any(np.abs(raw) >= 1):
        raise ValueError('atanh inverse requires raw commands strictly inside (-1,1)')
    logits = np.arctanh(raw).reshape(*raw.shape[:-2], 10)
    return full_to_params(logits, chart, atol=atol)


def raw_packet(coefficients, kind):
    """Single-packet public entry: coefficients (d,) -> raw time-major (5,2)."""
    coefficients = _params(coefficients, kind)
    if coefficients.ndim != 1:
        raise ValueError('raw_packet requires one coefficient vector, not a suffix/batch')
    return params_to_raw_packets(coefficients, kind)


def raw_mapping_jacobian(params, chart):
    """Per-packet d(flat raw commands)/d(params), shape (...,10,d).

    NOT the normalized model-action derivative: callers must additionally
    apply their frozen mean/std transform when differentiating that interface.
    """
    coefficients = _params(params, chart)
    raw = params_to_raw_packets(coefficients, chart).reshape(*coefficients.shape[:-1], 10)
    return (1 - raw ** 2)[..., :, None] * chart_matrix(chart)


def pullback_jacobian(j_full, chart, *, horizon=1):
    """J_chart = J_full @ C_H, where J_full = dF/d(full PRE-tanh logits).

    j_full shape (...,10*H); at nonzero u evaluate J_full at z=C_H u.
    A Jacobian wrt raw commands must first include diag(1-tanh(z)^2).
    """
    matrix = chart_matrix(chart, horizon=horizon)
    array = _array(j_full, 'j_full')
    if array.ndim < 1 or array.shape[-1] != matrix.shape[0] or array.size == 0:
        raise ValueError('full Jacobian requires trailing dimension 10*horizon')
    return array @ matrix


def pullback_gradient(g_full, chart, *, horizon=1):
    """Row-form gradient g_chart = g_full @ C_H (column form C_H.T @ g_full)."""
    return pullback_jacobian(g_full, chart, horizon=horizon)


def ridge_step(jacobian, residual, *, damping, current=None):
    """Solve min_delta ||J delta - residual||^2 + damping*||current+delta||^2.

    residual MUST be target minus current output. Default current is zero.
    damping is lambda (not sqrt(lambda)); strictly positive, no norm clipping.
    Uses augmented least squares to avoid squaring the condition number.
    """
    jacobian = _array(jacobian, 'jacobian')
    residual = _array(residual, 'residual')
    damping = _nonnegative(damping, 'damping')
    if (jacobian.ndim != 2 or min(jacobian.shape) < 1 or
            residual.shape != (jacobian.shape[0],) or damping <= 0):
        raise ValueError('nonempty 2D Jacobian, matching residual, positive damping required')
    width = jacobian.shape[1]
    current = np.zeros(width) if current is None else _array(current, 'current')
    if current.shape != (width,):
        raise ValueError('current must match the Jacobian coordinate width')
    regularizer = np.sqrt(damping)
    with np.errstate(over='ignore', invalid='ignore'):
        augmented = np.vstack((jacobian, regularizer * np.eye(width)))
        rhs = np.concatenate((residual, -regularizer * current))
    if not np.isfinite(rhs).all():
        raise ValueError('augmented ridge right-hand side overflow')
    step = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    if not np.isfinite(step).all():
        raise ValueError('ridge solve returned nonfinite step')
    return step


def chart_ridge_step(j_full, residual, chart, *, damping, horizon=1, current=None):
    """Ridge step in chart coordinates; constant is a RESTRICTED full solve.

    current, when supplied, is a flat d*H vector in this chart. This generally
    differs from projecting an unconstrained full ridge optimum afterward.
    """
    return ridge_step(pullback_jacobian(j_full, chart, horizon=horizon), residual,
                      damping=damping, current=current)
