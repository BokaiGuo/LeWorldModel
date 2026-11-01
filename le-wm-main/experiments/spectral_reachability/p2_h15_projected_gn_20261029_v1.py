"""CPU-side fixed-budget projected Gauss-Newton protocol, not a HOST runner.

Callbacks are deliberately abstract: a native Jacobian at an arbitrary logit has
not yet been connected. This module cannot authorize CUDA or physical execution.
"""
import numpy as np

RADIUS = .75 * np.sqrt(15)
DAMPING = .01
STEPS = (1., .5, .25, .125)
ITERATIONS = 4


def project(z):
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (30,) or not np.isfinite(z).all():
        raise ValueError('expected finite 30-dimensional logit')
    n = np.linalg.norm(z)
    return z * min(1., RADIUS / max(n, 1e-300))


def raw(z):
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (30,) or not np.isfinite(z).all() or np.linalg.norm(z) > RADIUS + 1e-12:
        raise ValueError('outside H15 logit ball')
    return np.tanh(z).astype(np.float32).reshape(15, 2)


def solve(goal, evaluate, linearize):
    """Four GN updates; 17 paid evaluations, four paid linearizations.

    evaluate(raw15) returns terminal latent. linearize(z) returns terminal
    latent and its 192x30 derivative with respect to *logits*. Callbacks must
    meter all internal predictor/AD work; these counts are not FLOPs.
    """
    goal = np.asarray(goal, np.float64)
    if goal.shape != (192,) or not np.isfinite(goal).all():
        raise ValueError('invalid identity192 goal')

    def terminal(value):
        a = np.asarray(value, np.float64)
        if a.shape != (192,) or not np.isfinite(a).all():
            raise ValueError('invalid terminal')
        return a

    z = np.zeros(30, np.float64)
    value = terminal(evaluate(raw(z)))
    score = float(np.mean((value-goal)**2))
    history = []
    for iteration in range(ITERATIONS):
        nominal, jac = linearize(z.copy())
        nominal = terminal(nominal)
        jac = np.asarray(jac, np.float64)
        if jac.shape != (192, 30) or not np.isfinite(jac).all():
            raise ValueError('invalid logit Jacobian')
        if not np.array_equal(nominal, value):
            raise ValueError('linearization must reproduce incumbent terminal')
        direction = np.linalg.solve(jac.T @ jac + DAMPING*np.eye(30), jac.T @ (goal-nominal))
        candidates = []
        for step in STEPS:
            candidate = project(z + step*direction)
            result = terminal(evaluate(raw(candidate)))
            q = float(np.mean((result-goal)**2))
            candidates.append((candidate, result, q))
        # First tie wins; incumbent survives a tie. All four candidates are paid.
        chosen = -1
        for i, (candidate, result, q) in enumerate(candidates):
            if q < score:
                z, value, score, chosen = candidate, result, q, i
        history.append(dict(iteration=iteration, scores=[x[2] for x in candidates], chosen=chosen, score=score))
    return dict(logits=z.tolist(), actions=raw(z).tolist(), score=score, history=history,
                evaluation_calls=17, linearization_calls=4)
