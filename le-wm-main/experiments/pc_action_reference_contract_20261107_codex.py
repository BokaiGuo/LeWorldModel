"""Distinguish a relative-action reference from a witnessed position hold."""
from dataclasses import dataclass
import hashlib

import numpy as np

from experiments.pc_action_witness_v2_20261107_codex import ActionWitnessV2, describe_bound


@dataclass(frozen=True)
class PositionHoldWitness:
    root_id: str
    clock: str
    action_fingerprint: str
    trajectory_sha256: str
    source_sha256: str
    steps: int
    max_position_displacement: float
    tolerance: float


def certify_position_hold(reference, positions, *, tolerance, source_sha256):
    """Evaluation-only evidence for one actual root and full observed trajectory."""
    if not isinstance(reference, ActionWitnessV2):
        raise TypeError("root- and clock-bound reference required")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("source sha256 required")
    tolerance = float(tolerance)
    positions = np.asarray(positions, dtype="<f8")
    if (not np.isfinite(tolerance) or tolerance < 0 or positions.ndim != 2
            or positions.shape[0] != len(reference.raw_actions) + 1
            or positions.shape[1] < 1 or not np.isfinite(positions).all()):
        raise ValueError("complete finite position trajectory and tolerance required")
    max_displacement = float(np.linalg.norm(positions - positions[0], axis=1).max())
    if max_displacement > tolerance:
        raise ValueError("reference is not a position hold on this observed trajectory")
    return PositionHoldWitness(reference.root_id, reference.clock, reference.fingerprint,
                               hashlib.sha256(positions.tobytes()).hexdigest(), source_sha256,
                               len(reference.raw_actions), max_displacement, tolerance)


def describe_reference(score, reference, probes, *, position_hold=None):
    """Always report relative effects; hold labeling requires scoped evidence."""
    if not isinstance(reference, ActionWitnessV2):
        raise TypeError("root- and clock-bound reference required")
    if position_hold is not None:
        if not isinstance(position_hold, PositionHoldWitness):
            raise TypeError("position-hold witness required")
        if ((position_hold.root_id, position_hold.clock, position_hold.action_fingerprint,
             position_hold.steps) != (reference.root_id, reference.clock, reference.fingerprint,
                                      len(reference.raw_actions))):
            raise ValueError("position-hold witness does not bind this reference")
    result = describe_bound(score, reference, probes)
    result["reference_semantics"] = "witnessed_position_hold" if position_hold else "relative_action_reference"
    result["position_hold_witness"] = position_hold
    return result
