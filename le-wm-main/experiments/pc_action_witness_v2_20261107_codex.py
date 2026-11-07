"""Immutable, root- and clock-bound float32 action witnesses."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True, init=False)
class ActionWitnessV2:
    name: str
    root_id: str
    clock: str
    raw_actions: tuple

    def __init__(self, name, root_id, clock, raw_actions):
        if any(not isinstance(x, str) or not x for x in (name, root_id, clock)):
            raise ValueError("name, root and clock must be explicit nonempty strings")
        actions = np.asarray(raw_actions, dtype=np.float32)
        if (actions.ndim != 2 or not actions.shape[0] or not actions.shape[1]
                or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
            raise ValueError("finite nonempty bounded time-by-action witness required")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "raw_actions", tuple(tuple(float(x) for x in row) for row in actions))

    def array(self):
        return np.asarray(self.raw_actions, dtype=np.float32)

    @property
    def fingerprint(self):
        actions = self.array().astype("<f4", copy=False)
        metadata = json.dumps({"root": self.root_id, "clock": self.clock, "shape": actions.shape,
                               "dtype": "<f4"}, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(metadata + b"\0" + actions.tobytes()).hexdigest()


def describe_bound(score, hold, probes):
    """Reject mixed roots/clocks before invoking any model query."""
    from experiments.pc_action_effect_descriptor import describe

    probes = tuple(probes)
    if not all(isinstance(x, ActionWitnessV2) for x in (hold, *probes)):
        raise TypeError("bound witnesses required")
    if any((p.root_id, p.clock) != (hold.root_id, hold.clock) for p in probes):
        raise ValueError("witness root/clock mismatch")
    return describe(score, hold, probes)
