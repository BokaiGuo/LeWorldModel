"""Fail-closed provenance and budget binding for a carry-plan feedback node.

This is an integration primitive, not a physical execution or fairness claim.
"""
from dataclasses import dataclass
import hashlib

import numpy as np


def _packet_array(value, *, ndim):
    array = np.asarray(value)
    if (array.dtype != np.float32 or array.ndim != ndim
            or array.shape[-2:] != (5, 2) or not np.isfinite(array).all()
            or np.any(np.abs(array) > 1)):
        raise ValueError("expected finite bounded float32 action packets")
    return np.ascontiguousarray(array).copy()


def _digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def extra_cost(num_new, horizon_packets, *, has_carry):
    """Upper bound for the selector's metered replay at this node."""
    if (type(num_new) is not int or num_new < 1 or type(horizon_packets) is not int
            or horizon_packets < 1 or type(has_carry) is not bool):
        raise ValueError("invalid selector dimensions")
    count = num_new + int(has_carry)
    steps = count * horizon_packets
    return dict(action_encode_calls=steps, predict_calls=steps,
                predict_samples=steps, replay_calls=count,
                selection_evaluations=count, backward_calls=0)


@dataclass(frozen=True)
class BoundPlan:
    """Plan selected before execution; the issuer must preserve the full plan."""
    episode: str
    boundary: int
    plan: np.ndarray
    plan_sha256: str

    @classmethod
    def seal(cls, episode, boundary, plan):
        if not isinstance(episode, str) or not episode or type(boundary) is not int or boundary < 0:
            raise ValueError("invalid plan identity")
        array = _packet_array(plan, ndim=3)
        if array.shape[0] < 1:
            raise ValueError("empty plan")
        array.setflags(write=False)
        return cls(episode, boundary, array, _digest(array))

    def bind_feedback(self, *, episode, boundary, issued_packet, rgb_sha256):
        """Return an unexecuted tail only for the immediate observed successor."""
        if (episode != self.episode or type(boundary) is not int
                or boundary != self.boundary + 5):
            raise ValueError("episode or boundary mismatch")
        if (not isinstance(rgb_sha256, str) or len(rgb_sha256) != 64
                or any(c not in "0123456789abcdef" for c in rgb_sha256)):
            raise ValueError("missing current RGB digest")
        if _digest(_packet_array(self.plan, ndim=3)) != self.plan_sha256:
            raise ValueError("selected plan changed after sealing")
        issued = _packet_array(issued_packet, ndim=2)
        if issued.tobytes() != self.plan[0].tobytes():
            raise ValueError("issued packet differs from selected plan")
        tail = self.plan[1:].copy()
        return {"tail": tail, "previous_plan_sha256": self.plan_sha256,
                "issued_packet_sha256": _digest(issued),
                "rgb_sha256": rgb_sha256, "episode": episode, "boundary": boundary}
