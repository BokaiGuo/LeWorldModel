"""Bounded multi-nominal response candidates; not a complete visual planner."""
import numpy as np


def generate(response, target, nominals, meter, *, epsilon=0.05, damping=1e-3,
             max_delta=0.5, mode="ridge"):
    """Measure each nominal's full Jacobian and generate one bounded proposal.

    The caller owns observation legality, selection, suffix optimization, and
    total resource limits. Every response evaluation and decomposition is metered.
    """
    target = np.asarray(target, dtype=float)
    nominal_array = np.asarray(nominals, dtype=float)
    if (target.ndim != 1 or not target.size or not np.isfinite(target).all()
            or nominal_array.ndim != 3 or nominal_array.shape[1:] != (5, 2)
            or not len(nominal_array) or not np.isfinite(nominal_array).all()
            or not 0 < epsilon < 1 or np.any(np.abs(nominal_array) > 1 - epsilon)
            or not np.isfinite(damping) or damping <= 0
            or not 0 < max_delta <= 1 or mode not in ("ridge", "svd")):
        raise ValueError("invalid response candidate configuration")

    def measure(action):
        result = np.asarray(meter.call("prediction_samples", 1, response, action), dtype=float)
        if result.shape != target.shape or not np.isfinite(result).all():
            raise ValueError("invalid model response")
        return result

    bank = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
    evidence = []
    for index, nominal in enumerate(nominal_array):
        center = measure(nominal)
        jacobian = np.stack([
            (measure(nominal + epsilon * direction) - measure(nominal - epsilon * direction))
            / (2 * epsilon) for direction in np.eye(10).reshape(10, 5, 2)
        ], axis=1)
        residual = target - center
        if mode == "ridge":
            delta = meter.call("linear_solve", 1, np.linalg.solve,
                               jacobian.T @ jacobian + damping * np.eye(10),
                               jacobian.T @ residual)
        else:
            u, singular, vt = meter.call("svd", 1, np.linalg.svd, jacobian, False)
            delta = vt.T @ ((singular / (singular ** 2 + damping)) * (u.T @ residual))
        raw_delta = delta.copy()
        scale = max(float(np.max(np.abs(delta))), max_delta)
        delta = delta * (max_delta / scale)
        proposal = np.clip(nominal + delta.reshape(5, 2), -1, 1)
        bank.append({"name": f"{mode}_{index}", "prefix": proposal.tolist()})
        evidence.append({"nominal": nominal.tolist(), "center": center.tolist(),
                         "jacobian": jacobian.tolist(), "residual": residual.tolist(),
                         "raw_delta": raw_delta.tolist(), "bounded_delta": delta.tolist(),
                         "proposal": proposal.tolist()})
    return bank, evidence
