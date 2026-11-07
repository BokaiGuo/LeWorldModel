"""Finite-support diagnostics with explicit units, ties, and root grouping."""
from dataclasses import dataclass
import numpy as np


def finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def effects(costs):
    costs = finite(costs, "costs")
    if costs.ndim < 1 or costs.shape[-1] < 2:
        raise ValueError("reference and at least one nonreference action required")
    return costs[..., :1] - costs[..., 1:]


def pairwise_agreement(predicted, actual, truth_tolerance=1e-6, prediction_tolerance=1e-8):
    predicted, actual = finite(predicted, "predicted"), finite(actual, "actual")
    if predicted.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("matching action vectors required")
    if truth_tolerance < 0 or prediction_tolerance < 0:
        raise ValueError("nonnegative tolerances required")
    correct, tied, count = 0, 0, 0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            target, estimate = actual[i] - actual[j], predicted[i] - predicted[j]
            if abs(target) <= truth_tolerance:
                continue
            count += 1
            if abs(estimate) <= prediction_tolerance:
                tied += 1
            elif target * estimate > 0:
                correct += 1
    return {"comparable": count, "correct": correct, "prediction_ties": tied,
            "agreement": (correct + .5 * tied) / count if count else None}


def ranking_record(scores, physical_costs):
    scores, physical_costs = finite(scores, "scores"), finite(physical_costs, "physical_costs")
    if scores.ndim != 1 or scores.shape != physical_costs.shape or not len(scores):
        raise ValueError("matching nonempty vectors required")
    choice = int(np.argmin(scores))
    return {"choice": choice, "regret": float(physical_costs[choice] - physical_costs.min()),
            **pairwise_agreement(scores, physical_costs)}


def unit_rows(features):
    features = finite(features, "features")
    if features.ndim != 2 or not len(features):
        raise ValueError("nonempty [sample, feature] matrix required")
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero-norm feature cannot be normalized")
    return features / norm


def squared_distances(x, y):
    x, y = finite(x, "x"), finite(y, "y")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("matching feature dimensions required")
    return np.maximum(np.sum(x*x, 1)[:, None] + np.sum(y*y, 1)[None] - 2 * (x @ y.T), 0)


@dataclass
class OrthogonalMap:
    source_mean: np.ndarray
    target_mean: np.ndarray
    rotation: np.ndarray
    source_rank: int
    cross_rank: int
    fit_rows: int
    calibration_roots: tuple

    def apply(self, features):
        return (finite(features, "features") - self.source_mean) @ self.rotation + self.target_mean


def fit_orthogonal(source, target, calibration_roots, evaluation_roots):
    """Row-vector affine Procrustes; disjoint roots, no outcome labels accepted."""
    source, target = finite(source, "source"), finite(target, "target")
    calibration_roots, evaluation_roots = tuple(calibration_roots), tuple(evaluation_roots)
    if not calibration_roots or not evaluation_roots or set(calibration_roots) & set(evaluation_roots):
        raise ValueError("nonempty, disjoint root groups required")
    if source.ndim != 2 or source.shape != target.shape or len(source) < 2:
        raise ValueError("paired feature matrices required")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    cross = x.T @ y
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return OrthogonalMap(source_mean, target_mean, u @ vt, int(np.linalg.matrix_rank(x)),
                         int(np.linalg.matrix_rank(cross)), len(source), calibration_roots)


def exact_readout_possible(features, descriptors):
    """Finite-domain sufficiency only, exact equality and no generalization claim."""
    features, descriptors = finite(features, "features"), finite(descriptors, "descriptors")
    if features.ndim != 2 or descriptors.ndim != 2 or len(features) != len(descriptors):
        raise ValueError("paired feature and descriptor matrices required")
    for i in range(len(features)):
        for j in range(i):
            if np.array_equal(features[i], features[j]) and not np.array_equal(descriptors[i], descriptors[j]):
                return False
    return True


def aliasing_bound(delta, feature_distance, lipschitz):
    values = finite([delta, feature_distance, lipschitz], "bound arguments")
    if np.any(values < 0):
        raise ValueError("bound arguments must be nonnegative")
    return float(max(0, delta - lipschitz * feature_distance) / 2)


def finite_selection_bound(predicted, truth):
    """Requires costs in identical units over the identical ordered support."""
    predicted, truth = finite(predicted, "predicted"), finite(truth, "truth")
    if predicted.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError("identical nonempty support required")
    epsilon = float(np.max(np.abs(predicted - truth)))
    choice = int(np.argmin(predicted))
    return {"epsilon": epsilon, "regret": float(truth[choice] - truth.min()), "bound": 2 * epsilon}


def root_summary(values, seed=71107, samples=2000):
    """Descriptive root bootstrap, not proof of IID or source independence."""
    values = finite(values, "root values")
    if values.ndim != 1 or not len(values):
        raise ValueError("nonempty root values required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    return {"n_roots": len(values), "mean": float(values.mean()),
            "bootstrap95_descriptive": np.quantile(means, [.025, .975]).tolist()}
