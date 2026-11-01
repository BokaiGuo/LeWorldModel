"""Development-only, fail-closed RGB approach interface; does not run physics.

One action is proposed per verified-in-sequence observation receipt. The caller
must supply a real post-action RGB observation, but this module cannot prove
camera provenance, execution, or physical safety from caller assertions.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import codex_p1_rgb_task_score_20261030_v2 as rgb
from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry
from . import p1_dev302_mixed_prefix_bank_v5_20261101_codex as motion
from . import p1_dev302_rgb_contact_feedback_20261101_codex as prior

NAME = "p1_rgb_chunk_approach_interface_20261102_exclusive"
MAX_CHUNKS = 5


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _distance_segment(start, end, pixels):
    direction = end - start
    squared = float(direction @ direction)
    if squared == 0.0:
        return float(np.min(np.linalg.norm(pixels - start, axis=1)))
    fraction = np.clip(((pixels - start) @ direction) / squared, 0.0, 1.0)
    closest = start + fraction[:, None] * direction
    return float(np.min(np.linalg.norm(pixels - closest, axis=1)))


class RGBChunkApproach:
    """In-memory proposal/receipt state machine, not an execution authority.

    A trusted integration must establish that the starting prior action was
    actually zero and that each receipt refers to the immediately preceding
    action and its subsequently captured RGB. Receipt fields are assertions,
    not authentication. A failure permanently closes this instance.
    """

    def __init__(self, *, initial_previous_action_is_zero):
        _require(initial_previous_action_is_zero is True, "zero prior action assertion required")
        self._previous = np.zeros(2, dtype=np.float64)
        self._pending = None
        self._last_frame = None
        self._chunks = 0
        self._closed = False

    def observe(self, image, *, frame_id, executed_action_sha256=None):
        """Accept exactly one RGB receipt, and return one proposal or stop.

        The caller must send the action hash returned by the immediately prior
        proposal, and a strictly increasing integer frame id. No action is sent
        by this API. Any malformed/missing receipt causes permanent abstention.
        """
        if self._closed:
            raise ValueError("controller closed")
        try:
            _require(type(frame_id) is int and frame_id >= 0, "integer frame id")
            _require(self._last_frame is None or frame_id > self._last_frame,
                     "fresh frame receipt required")
            if self._pending is None:
                _require(self._last_frame is None and executed_action_sha256 is None,
                         "unexpected or missing action receipt")
            else:
                _require(executed_action_sha256 == self._pending["action_sha256"],
                         "previous action receipt mismatch")
                self._previous = np.asarray(self._pending["action"], dtype=np.float64)
                self._pending = None
            image = rgb._image(image)
            _, pusher = rgb._mask(image, rgb.BLUE, 8, single=False)
            object_mask, obj = rgb._mask(image, rgb.BLOCK, 8, single=False)
            _require(obj["components"] == 1, "fragmented object segmentation")
            x0, y0, x1, y1 = pusher["bounds_xyxy"]
            _require(pusher["components"] <= prior.MOVING_PUSHER_COMPONENTS_MAX
                     and x1 - x0 + 1 <= prior.MOVING_PUSHER_BBOX_MAX_PIXELS
                     and y1 - y0 + 1 <= prior.MOVING_PUSHER_BBOX_MAX_PIXELS,
                     "ambiguous pusher segmentation")
            self._last_frame = frame_id
            position = np.asarray(pusher["centroid_xy"], dtype=np.float64)
            py, px = np.nonzero(object_mask)
            pixels = np.column_stack((px, py)).astype(np.float64)
            vectors = pixels - position
            distances = np.linalg.norm(vectors, axis=1)
            index = int(np.argmin(distances))
            gap = float(distances[index])
            _require(math.isfinite(gap), "finite gap")
            if gap <= prior.TARGET_GAP_PIXELS:
                self._closed = True
                return self._stop("rgb_near_object_no_contact_claim", frame_id)
            if self._chunks >= MAX_CHUNKS:
                self._closed = True
                return self._stop("chunk_budget_exhausted", frame_id)
            direction = vectors[index] / gap
            magnitude = min(prior.MAX_RAW_MAGNITUDE,
                            (gap - prior.TARGET_GAP_PIXELS)
                            * prior.NATIVE_PER_PIXEL / prior.RAW_TARGET_SCALE_NATIVE)
            action = np.asarray(direction * magnitude, dtype="<f4")
            _require(np.isfinite(action).all()
                     and np.linalg.norm(action.astype(np.float64)) <= prior.MAX_RAW_MAGNITUDE + 1e-7,
                     "bounded action")
            delta = (
                motion.NATIVE_DELTA_PER_CURRENT_ACTION * action.astype(np.float64)
                + motion.NATIVE_DELTA_PER_PREVIOUS_VELOCITY
                * motion.NATIVE_VELOCITY_PER_PREVIOUS_ACTION * self._previous
            ) * motion.PIXELS_PER_NATIVE
            end = position + delta
            radius = geometry.PUSHER_RADIUS_PIXELS
            height, width = image.shape[:2]
            for point in (position, end):
                _require(radius <= point[0] <= width - 1 - radius
                         and radius <= point[1] <= height - 1 - radius,
                         "predicted pusher disk outside image")
            _require(_distance_segment(position, end, pixels) >= prior.TARGET_GAP_PIXELS,
                     "predicted approach crosses conservative object margin")
            digest = hashlib.sha256(action.tobytes()).hexdigest()
            self._pending = {"action": action.tolist(), "action_sha256": digest}
            self._chunks += 1
            return {
                "schema": NAME + ".proposal.v1",
                "status": "proposal_only_wait_for_real_rgb",
                "frame_id": frame_id,
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "chunk": self._chunks,
                "action": action.tolist(),
                "action_sha256": digest,
                "predicted_next_center_xy": end.tolist(),
                "receipt_required_before_next_proposal": True,
                "physical_execution_authorized": False,
                "P1_admission": False,
                "independent_confirmation": False,
            }
        except Exception:
            self._closed = True
            self._pending = None
            raise

    @staticmethod
    def _stop(reason, frame_id):
        return {"schema": NAME + ".stop.v1", "status": reason, "frame_id": frame_id,
                "physical_execution_authorized": False, "P1_admission": False,
                "independent_confirmation": False}
