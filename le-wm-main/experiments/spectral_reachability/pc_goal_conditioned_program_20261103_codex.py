"""Goal-conditioned candidate generation and complete-chain planning.

This is a development entry point. Callers provide only deployable model
features, a goal expressed in the same weighted feature coordinates, and a
metered suffix optimizer. It does not own observation or physical execution.
"""
import numpy as np

from experiments.spectral_reachability.d2p_goal_mode_rank_20261102 import rank_goal_modes
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import plan_bank


def generate(response, target, meter, *, mode, seed, damping=1e-3,
             epsilon=.05, amplitude=.5):
    """Generate bounded five-step candidates with fixed response coordinates."""
    target = np.asarray(target, dtype=float)
    if (target.ndim != 1 or not target.size or not np.isfinite(target).all()
            or mode not in ("goal", "singular", "random", "ridge")
            or not 0 < epsilon <= 1 or not 0 < amplitude <= 1
            or not np.isfinite(damping) or damping <= 0):
        raise ValueError("invalid target or candidate policy")
    zero = np.zeros((5, 2), dtype=float)

    def measured(action):
        value = np.asarray(meter.call("prediction_samples", 1, response, action), dtype=float)
        if value.shape != target.shape or not np.isfinite(value).all():
            raise ValueError("response and target coordinates differ")
        return value

    center = measured(zero)
    columns = []
    for direction in np.eye(10).reshape(10, 5, 2):
        columns.append((measured(epsilon * direction) - measured(-epsilon * direction))
                       / (2 * epsilon))
    jacobian = np.stack(columns, axis=1)
    residual = target - center
    evidence = {"mode": mode, "jacobian": jacobian.tolist(),
                "residual": residual.tolist(), "damping": damping}
    if mode in ("goal", "singular"):
        ranked = meter.call("svd", 1, rank_goal_modes, jacobian, residual, damping)
        index = int(ranked["order"][0]) if mode == "goal" else 0
        if ranked["singular"][index] <= 1e-10:
            direction = np.zeros(10)
        else:
            direction = ranked["right_t"][index]
        evidence.update({"selected_mode": index,
                         "singular_values": ranked["singular"].tolist(),
                         "goal_contributions": ranked["contributions"].tolist()})
    elif mode == "ridge":
        def solve():
            return np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(10),
                                   jacobian.T @ residual)
        direction = meter.call("linear_solve", 1, solve)
    else:
        direction = np.random.default_rng(seed).normal(size=10)
    bank = [{"name": "zero", "prefix": zero.tolist(), "source": "model"}]
    scale = np.max(np.abs(direction))
    if scale > 1e-12:
        for sign in (-1, 1):
            candidate = (sign * amplitude * direction / scale).reshape(5, 2)
            bank.append({"name": f"{mode}_{sign:+d}", "prefix": candidate.tolist(),
                         "source": "random" if mode == "random" else "model"})
    return bank, evidence


def compile_program(response, target, prefix_score, optimize_suffix, meter, *,
                    mode, seed, initial_score, suffix_samples, damping=1e-3):
    """Generate candidate prefixes then choose using bounded complete rollouts."""
    bank, evidence = generate(response, target, meter, mode=mode, seed=seed,
                              damping=damping)
    result = plan_bank(bank, initial_score, prefix_score, optimize_suffix,
                       meter, suffix_samples)
    result.update({"generator": evidence, "mode": mode})
    return result
