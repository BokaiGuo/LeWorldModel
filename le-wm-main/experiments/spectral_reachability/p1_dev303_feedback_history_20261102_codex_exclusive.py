"""Chronological RGB/action receipts for prospective dev303 feedback; no execution."""
from __future__ import annotations

import hashlib

import numpy as np

from . import p1_dev303_rgb_feedback_scored_tail_20261102_codex_exclusive as feedback


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class History:
    """A trusted runner must call reset once and append after each real env.step.

    Captures are copied and checked; provenance itself is the runner's job.
    Any inconsistent step/receipt permanently closes this instance.
    """
    def __init__(self):
        self.frames = []
        self.actions = []
        self.frame_ids = []
        self.pending = None
        self.closed = False

    def reset(self, image, frame_id):
        try:
            feedback.require(not self.closed and not self.frames and type(frame_id) is int
                             and frame_id == 0, "history reset")
            self.frames.append(feedback.rgb._image(image).copy())
            self.frame_ids.append(frame_id)
        except Exception:
            self.closed = True
            raise

    def dispatched(self, action, *, step):
        try:
            feedback.require(not self.closed and self.frames and self.pending is None
                             and type(step) is int and step == self.frame_ids[-1] + 1,
                             "history dispatch sequence")
            self.pending = feedback.raw_actions([action], (1, 2))[0].copy()
            return digest(self.pending.tobytes())
        except Exception:
            self.closed = True
            raise

    def returned(self, actual_action, image, *, step):
        try:
            feedback.require(not self.closed and self.pending is not None and type(step) is int
                             and step == self.frame_ids[-1] + 1, "history return sequence")
            actual = feedback.raw_actions([actual_action], (1, 2))[0]
            feedback.require(np.array_equal(actual, self.pending), "history actual action mismatch")
            pixels = feedback.rgb._image(image).copy()
            self.actions.append(actual.copy())
            self.frames.append(pixels)
            self.frame_ids.append(step)
            self.pending = None
        except Exception:
            self.closed = True
            raise

    def context(self):
        feedback.require(not self.closed and self.pending is None and len(self.actions) >= 10,
                         "incomplete actual history")
        frames = [p.copy() for p in self.frames[-11:]]
        actions = np.asarray(self.actions[-10:], dtype="<f4")
        ids = self.frame_ids[-11:]
        feedback.require(ids == list(range(ids[0], ids[0] + 11)), "non-consecutive history")
        return {"frames": frames, "actions": actions.copy(), "frame_ids": ids.copy(),
                "pixel_sha256": [digest(x.tobytes()) for x in frames],
                "actual_history_sha256": digest(actions.tobytes())}
