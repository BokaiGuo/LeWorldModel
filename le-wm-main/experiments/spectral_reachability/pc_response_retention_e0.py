"""Fixed-bank retention rules for a successor-response development diagnostic."""
import numpy as np


def select(scores, states, responses, ridge, lookahead, seed, count=4):
    scores = np.asarray(scores, np.float64)
    states, responses, ridge, lookahead = (np.asarray(x, np.float64)
                                            for x in (states, responses, ridge, lookahead))
    n = len(scores)
    if (scores.shape != (n,) or states.shape[0] != n or responses.shape[0] != n
            or ridge.shape != (n,) or lookahead.shape != (n,) or count < 2 or count > n
            or not all(np.isfinite(x).all() for x in (scores, states, responses, ridge, lookahead))):
        raise ValueError("invalid fixed-bank retention inputs")
    greedy = np.argsort(scores, kind="stable")[:count].tolist()

    def farthest(chosen, features):
        features = features.reshape(n, -1)
        while len(chosen) < count:
            distances = np.linalg.norm(features[:, None] - features[chosen][None], axis=2).min(axis=1)
            distances[chosen] = -np.inf
            chosen.append(int(np.argmax(distances)))
        return chosen

    state = farthest(greedy[:2], states)
    response = [greedy[0]]
    for index in np.argsort(ridge, kind="stable"):
        if int(index) not in response:
            response.append(int(index))
            break
    response = farthest(response, responses)
    random = np.random.default_rng(seed).choice(n, size=count, replace=False).tolist()
    return {"greedy": greedy, "state": state, "response": response,
            "lookahead": np.argsort(lookahead, kind="stable")[:count].tolist(), "random": random}


def ridge_residual(response, residual, damping=.1):
    matrix, target = np.asarray(response, np.float64), np.asarray(residual, np.float64)
    if (matrix.ndim != 2 or target.shape != (matrix.shape[0],) or damping <= 0
            or not np.isfinite(matrix).all() or not np.isfinite(target).all()):
        raise ValueError("invalid ridge inputs")
    weights = matrix.T @ np.linalg.solve(matrix @ matrix.T + damping * np.eye(len(target)), target)
    error = target - matrix @ weights
    return float(error @ error + damping * (weights @ weights))
