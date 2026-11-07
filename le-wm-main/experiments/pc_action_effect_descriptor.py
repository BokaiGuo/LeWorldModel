"""Action-indexed finite-support consequence vectors; no hidden-state access."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch


@dataclass(frozen=True)
class ActionWitness:
    name: str
    raw_actions: tuple

    def array(self):
        value = np.asarray(self.raw_actions, dtype=np.float64)
        if value.ndim != 2 or not value.shape[0] or not value.shape[1] or not np.isfinite(value).all():
            raise ValueError("finite nonempty time-by-action witness required")
        if np.any(np.abs(value) > 1):
            raise ValueError("raw action outside declared [-1,1] domain")
        return value

    @property
    def fingerprint(self):
        value = self.array().astype("<f8")
        return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def describe(score, hold, probes):
    """Positive effect means improvement relative to the equal-duration hold."""
    shape = hold.array().shape
    names = [hold.name] + [p.name for p in probes]
    if len(set(names)) != len(names):
        raise ValueError("ambiguous action identities")
    if any(p.array().shape != shape for p in probes):
        raise ValueError("probe duration/action dimension mismatch")
    costs = torch.stack([score(p.array()) for p in (hold, *probes)])
    if costs.ndim != 1 or not torch.isfinite(costs).all():
        raise ValueError("score must return a finite scalar per witness")
    return {"costs": costs, "effects": costs[0] - costs[1:],
            "actions": [{"name": p.name, "fingerprint": p.fingerprint,
                         "raw_actions": p.array().tolist()} for p in (hold, *probes)],
            "queries": len(names)}


def central_derivative(minus, plus, epsilon):
    if epsilon <= 0:
        raise ValueError("positive raw-action displacement required")
    return (plus - minus) / (2 * epsilon)


def quadratic_prediction(parent, minus, plus, epsilon, displacements):
    slope = central_derivative(minus, plus, epsilon)
    curvature = (plus + minus - 2 * parent) / epsilon**2
    return parent + slope * displacements + .5 * curvature * displacements**2
