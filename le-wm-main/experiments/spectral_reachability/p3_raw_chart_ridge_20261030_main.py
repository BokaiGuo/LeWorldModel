"""Exploratory raw-coordinate ridge solver; not connected to a P3 gate.

The Jacobian, the optimized variable, and the returned action share the raw
action chart. This is an offline planning primitive, not neural inference.
"""
import math

import numpy as np
from scipy.optimize import minimize


def solve_raw_ridge(jacobian, residual, *, damping=0.002, radius=math.sqrt(5) * 0.75):
    """Minimize squared linear residual plus ridge under a box and L2 ball.

    Returns the actual float32 raw packet and recomputes the objective after
    conversion; never applies tanh. Failure to solve or satisfy bounds raises.
    """
    j = np.asarray(jacobian, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if (j.ndim != 2 or j.shape[1] != 10 or j.shape[0] == 0
            or r.shape != (j.shape[0],) or not np.isfinite(j).all()
            or not np.isfinite(r).all() or not math.isfinite(damping)
            or damping <= 0 or not math.isfinite(radius) or radius <= 0):
        raise ValueError("invalid raw ridge inputs")

    def objective(x):
        error = r - j @ x
        return float((error @ error + damping * (x @ x)) / r.size)

    def gradient(x):
        return 2 * (j.T @ (j @ x - r) + damping * x) / r.size

    result = minimize(
        objective, np.zeros(10), jac=gradient, method="SLSQP",
        bounds=[(-1.0, 1.0)] * 10,
        constraints=[{"type": "ineq", "fun": lambda x: radius**2 - x @ x,
                      "jac": lambda x: -2 * x}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"raw ridge solver failed: {result.message}")
    raw = np.asarray(result.x, dtype=np.float32)
    if np.max(np.abs(raw)) > 1 or np.linalg.norm(raw.astype(np.float64)) > radius + 1e-6:
        raise RuntimeError("raw ridge solver produced an infeasible packet")
    score = objective(raw.astype(np.float64))
    if score > objective(np.zeros(10)) + 1e-10:
        raise RuntimeError("raw ridge solver worsened zero initialization")
    return {"raw": raw.reshape(5, 2), "linear_objective": score,
            "iterations": int(result.nit), "radius": float(radius),
            "coordinate": "raw_float32", "optimizer_success": True}
