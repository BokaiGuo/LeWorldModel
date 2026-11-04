"""Exploratory nonlinear model-consistency guard for spectral proposals."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def guarded_candidates(terminal, origin, goal, *, modes=2, damping=0.1,
                       scales=(0.5, 0.1, 0.02), tolerance=0.3, min_signal=1e-5):
    """Compile signed modes with individually charged nonlinear backtracking.

    This verifies local agreement with the *same model*, not with physics.
    The zero and DLS controls are included but are not certified by this guard.
    """
    if (origin.ndim != 1 or goal.ndim != 1 or modes < 1 or damping <= 0
            or not 0 < tolerance < 1 or min_signal <= 0 or not scales
            or any(s <= 0 for s in scales)
            or any(scales[i] <= scales[i + 1] for i in range(len(scales) - 1))):
        raise ValueError("invalid guard contract")
    point = origin.detach().requires_grad_(True)
    base = terminal(point)
    if base.shape != goal.shape or not torch.isfinite(base).all() or not torch.isfinite(goal).all():
        raise ValueError("invalid goal or model output")
    jac = torch.autograd.functional.jacobian(terminal, point, vectorize=True)
    if jac.ndim != 2 or not torch.isfinite(jac).all() or jac.norm() <= 1e-10:
        raise ValueError("invalid Jacobian")
    left, singular, right = torch.linalg.svd(jac, full_matrices=False)
    count = min(modes, len(singular))
    indices = list(dict.fromkeys(list(range(count)) + list(range(len(singular) - count, len(singular)))))
    residual = goal - base
    dls = right.T @ ((singular / (singular.square() + damping)) * (left.T @ residual))
    dls = dls * min(1.0, scales[0] / max(float(dls.norm().detach()), 1e-12))
    bank = [origin.detach(), (origin + dls).detach()]
    candidate_labels = ["neutral", "dls_unchecked"]
    probes = []
    with torch.no_grad():
        for mode in indices:
            for sign in (1, -1):
                direction = sign * right[mode]
                tangent = jac @ direction
                for scale in scales:
                    actual = terminal(origin + scale * direction) - base
                    signal = float(actual.norm())
                    error = float((actual - scale * tangent).norm()) / max(signal, 1e-12)
                    accepted = signal >= min_signal and error <= tolerance
                    probes.append({"mode": mode, "sign": sign, "scale": scale,
                                   "signal": signal, "relative_error": error, "accepted": accepted})
                    if accepted:
                        bank.append((origin + scale * direction).detach())
                        candidate_labels.append(f"mode_{mode}_sign_{sign}_scale_{scale}")
                        break
    return torch.stack(bank), {
        "candidate_labels": candidate_labels, "probes": probes,
        "candidate_count": len(bank), "mode_indices": indices,
        "singular_values": singular.detach().cpu().tolist(),
        "backward_vjp_rows": int(jac.shape[0]),
        "baseline_and_jacobian_primal_rollouts": 2,
        "nonlinear_guard_primal_rollouts": len(probes),
        "limits": "DLS unchecked; local model consistency is not physical evidence",
    }


def compile_reacher(model, pixels, goal, history, *, lower, upper):
    if history.shape[0] != 1:
        raise ValueError("exactly one root required")
    dimension = history.shape[-1]
    goal_coordinate = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()
    def terminal(flat):
        value = p0._terminal_batch(model, pixels, goal, history, flat.reshape(1, 1, dimension),
                                   action_lower=lower, action_upper=upper)[0]
        return p0._planning_coordinate(model, value, role="predicted")
    origin = torch.zeros(dimension, dtype=history.dtype, device=history.device)
    bank, receipt = guarded_candidates(terminal, origin, goal_coordinate)
    return bank[:, None, :], receipt
