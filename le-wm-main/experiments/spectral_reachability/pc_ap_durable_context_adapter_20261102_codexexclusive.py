"""Durable five-step feedback adapter; no environment creation or model calls.

Caller owns the host lock, input provenance and planner. Never resume a journal
after a crash: an unmatched intent may represent a physical action.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import pc_ap_feedback_step_bridge_20261102_codexexclusive as bridge
from . import reacher_d2p_context as runtime


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


class Journal:
    """New exclusive journal; each complete line is flushed and fsynced."""
    def __init__(self, path):
        path = Path(path)
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError("absolute path with existing parent required")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.file = os.fdopen(fd, "wb", buffering=0)
        self.previous = None
        self.index = 0
        self.failed = False
        self.claimed = False
        try:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            self.failed = True
            self.file.close()
            raise

    def emit(self, payload):
        if self.failed:
            raise RuntimeError("journal poisoned")
        event = {"index": self.index, "previous_sha256": self.previous, "payload": payload}
        event["event_sha256"] = _digest(event)
        line = _encoded(event) + b"\n"
        try:
            if self.file.write(line) != len(line):
                raise OSError("short journal write")
            os.fsync(self.file.fileno())
        except BaseException:
            self.failed = True
            raise
        self.index += 1
        self.previous = event["event_sha256"]

    def close(self):
        self.file.close()


def _material(context):
    frames = [bridge._rgb(x) for x in context["frames"]]
    root = bridge._rgb(context["root_frame"])
    goal = bridge._rgb(context["goal"])
    blocks = [np.asarray(x) for x in context["history_action_blocks"]]
    if (len(frames) != 3 or len(blocks) != 3 or
            any(x.shape != root.shape for x in frames) or goal.shape != root.shape or
            any(x.dtype != np.float32 or x.shape != (10,) or not np.isfinite(x).all()
                for x in blocks)):
        raise ValueError("expected Reacher 3 RGB frames and three float32 5x2 blocks")
    if (context["root_rgb_sha256"] != runtime.rgb_digest(root) or
            context["goal_rgb_sha256"] != runtime.rgb_digest(goal) or
            context["history_rgb_sha256"] != [runtime.rgb_digest(x) for x in frames]):
        raise ValueError("context RGB hashes do not match pixels")
    snapshot = context["execution_snapshot"]
    if not np.array_equal(runtime.unpack_rgb_observation(snapshot["rendered_goal_rgb"]), goal):
        raise ValueError("snapshot goal RGB differs")
    packed = {"frames": [runtime.pack_rgb_observation(x) for x in frames],
              "root": runtime.pack_rgb_observation(root), "goal": runtime.pack_rgb_observation(goal),
              "history_action_float32_hex": [np.ascontiguousarray(x).tobytes().hex() for x in blocks],
              "execution_snapshot": snapshot}
    return frames, root, goal, [x.copy() for x in blocks], packed


class Adapter:
    """One owner, one journal, sequential completed blocks; no recovery."""
    def __init__(self, journal):
        if not isinstance(journal, Journal) or journal.claimed or journal.failed or journal.file.closed:
            raise ValueError("a new, unclaimed exclusive journal is required")
        journal.claimed = True
        self.journal = journal
        self.poisoned = False
        self.expected_context = None
        self.decision = 0

    def advance(self, env, context, actions):
        if self.poisoned:
            raise RuntimeError("adapter poisoned; never retry an uncertain dispatch")
        self.poisoned = True
        try:
            frames, root, goal, blocks, packed = _material(context)
            context_hash = _digest(packed)
            if self.expected_context is not None and context_hash != self.expected_context:
                raise ValueError("context does not match previous durable handoff")
            live = runtime.capture_execution_snapshot(env)
            runtime.assert_exact_execution_state(context["execution_snapshot"], live)
            self.journal.emit({"kind": "context_bound", "decision": self.decision,
                               "context_sha256": context_hash, "context": packed})

            def emit(event):
                # bridge's event precedes snapshot capture; only context_committed
                # below authorizes any next decision.
                self.journal.emit({"kind": "bridge_event", "decision": self.decision, "event": event})

            receipt, handoff = bridge.execute_block(
                env, actions, expected_root=root, expected_goal=context["goal_rgb_sha256"],
                goal=goal, emit=emit)
            if handoff is None:
                self.journal.emit({"kind": "block_incomplete", "decision": self.decision, "receipt": receipt})
                return receipt, None
            snapshot = runtime.capture_execution_snapshot(env)
            # Completeness and goal/scene presence are mandatory, even after
            # a successful render. Compare snapshot against itself to apply ABI.
            runtime.assert_exact_execution_state(snapshot, snapshot)
            returned = np.asarray(handoff["executed_actions"])
            step_bits = [item["action_float32_hex"] for item in receipt["steps"]]
            if (returned.dtype != np.float32 or returned.shape != (5, 2) or
                    step_bits != [x.tobytes().hex() for x in returned] or
                    receipt["attempted"] != 5 or receipt["returned"] != 5):
                raise RuntimeError("actual completed action history mismatch")
            successor = bridge._rgb(handoff["root_frame"])
            result = {
                "frames": frames[1:] + [root],
                "goal": goal,
                "history_action_blocks": blocks[1:] + [returned.reshape(-1).copy()],
                "execution_snapshot": snapshot,
                "root_frame": successor,
                "context_mode": "executed_warmup_plus_observed_action",
                "parent_context_mode": context.get("context_mode"),
                "root_identity_validation": "fresh_exact_integration_and_scene",
                "history_rgb_sha256": [runtime.rgb_digest(x) for x in frames[1:] + [root]],
                "root_rgb_sha256": runtime.rgb_digest(successor),
                "goal_rgb_sha256": runtime.rgb_digest(goal),
                "advanced_action_sha256": bridge._hash(returned),
            }
            _, _, _, _, next_packed = _material(result)
            next_hash = _digest(next_packed)
            self.journal.emit({"kind": "context_committed", "decision": self.decision,
                               "previous_context_sha256": context_hash, "context_sha256": next_hash,
                               "receipt": receipt, "context": next_packed})
            self.expected_context = next_hash
            self.decision += 1
            self.poisoned = False
            return receipt, result
        except BaseException:
            # Do not claim no action occurred or retry when persistence failed.
            raise
