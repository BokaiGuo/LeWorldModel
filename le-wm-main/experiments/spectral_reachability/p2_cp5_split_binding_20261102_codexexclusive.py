"""Experimental split-episode binding core, NOT a native physical launcher.

The session provider must independently enforce and record the original native
warmup/prefix guard. Until a verified native provider and saved-result consumer
exist, this module is suitable only for fake-session integration tests.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from . import p3_goal_patch_consistent_compiler_20261027_v2 as normalization


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def actions(value, n):
    a = np.asarray(value)
    need(a.dtype == np.dtype("<f4") and a.shape == (n, 2) and
         np.isfinite(a).all() and (np.abs(a) <= 1).all(), "invalid raw float32 actions")
    return np.ascontiguousarray(a)


class Adapter:
    """Capture callback ABI and strict replay gate over a fresh-session provider.

    A provider's reconstruct must return actual RGB/history and a nonempty
    independently checked guard; this class cannot authenticate that provider.
    The caller must authenticate and freeze model winner selection separately.
    """
    def __init__(self, path, factory, root_sha256, task_id, base_history, base_rgb_sha256):
        self.path = Path(path)
        need(not self.path.exists() and not self.path.is_symlink(), "output already exists")
        need(callable(factory) and re.fullmatch(r"[0-9a-f]{64}", root_sha256) and
             isinstance(task_id, str) and task_id, "binding")
        self.factory, self.root, self.task = factory, root_sha256, task_id
        self.history = np.asarray(base_history)
        need(self.history.dtype == np.dtype("<f4") and self.history.shape == (2, 10)
             and np.isfinite(self.history).all(), "base history")
        self.history = self.history.copy()
        need(len(base_rgb_sha256) == 2 and all(re.fullmatch(r"[0-9a-f]{64}", x)
             for x in base_rgb_sha256), "base RGB hashes")
        self.base_rgb = list(base_rgb_sha256)
        self.path.mkdir()
        self.receipts = {}
        self.failed = False
        self.steps_attempted = 0
        self.steps_returned = 0
        self.max_steps = 170  # 32 captures + two 5+5 replays, excludes reconstruction.

    def write(self, name, content):
        with (self.path / name).open("xb") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def start(self):
        session = self.factory()
        try:
            root = session.reconstruct()
            rgb = np.asarray(root["rgb"])
            h = np.asarray(root["history"])
            need(root["root_sha256"] == self.root and root["task_id"] == self.task
                 and isinstance(root["guard"], str) and root["guard"], "root/guard mismatch")
            need(rgb.dtype == np.uint8 and rgb.shape == (2, 224, 224, 3) and
                 [digest(x.tobytes()) for x in rgb] == self.base_rgb, "root RGB mismatch")
            need(h.dtype == np.dtype("<f4") and h.shape == (2, 10) and
                 h.tobytes() == self.history.tobytes(), "root history mismatch")
            return session, root
        except BaseException:
            session.close()
            raise

    def advance(self, session, seq):
        last = None
        for command in seq:
            need(self.steps_attempted < self.max_steps, "candidate step cap")
            self.steps_attempted += 1
            last = session.step(command.copy())
            self.steps_returned += 1
            need(type(last["terminated"]) is bool and type(last["truncated"]) is bool,
                 "stop flags")
            if last["terminated"] or last["truncated"]:
                raise RuntimeError("native stop: terminated=%s truncated=%s" %
                                   (last["terminated"], last["truncated"]))
        return last

    def capture(self, arm, index, prefix):
        need(not self.failed and re.fullmatch(r"[a-z_]+", arm or "") and
             type(index) is int and 0 <= index < 16 and (arm, index) not in self.receipts
             and len(self.receipts) < 32, "capture route")
        prefix = actions(prefix, 5)
        stem = f"capture_{arm}_{index:02d}"
        session = None
        try:
            self.write(stem + "_intent.json", canonical({"arm": arm, "index": index,
                "root": self.root, "task": self.task, "prefix_hex": prefix.tobytes().hex()}))
            session, root = self.start()
            last = self.advance(session, prefix)
            image = np.asarray(last["rgb"])
            need(image.dtype == np.uint8 and image.shape == (224, 224, 3), "cp5 RGB")
            rgb = np.concatenate((np.asarray(root["rgb"]), image[None]))
            history = np.stack((self.history[1], normalization.normalized_history(prefix)[0])).astype("<f4")
            record = {"arm": arm, "candidate_index": index, "root_sha256": self.root,
                      "task_id": self.task, "prefix_hex": prefix.tobytes().hex(),
                      "rgb_sha256": [digest(x.tobytes()) for x in rgb],
                      "action_history_hex": history.tobytes().hex(),
                      "guard": root["guard"], "snapshot": last["snapshot"]}
            raw = canonical(record)
            self.write(stem + "_rgb.bin", rgb.tobytes())
            self.write(stem + "_receipt.json", raw)
            receipt = digest(raw)
            self.receipts[(arm, index)] = (receipt, record)
            return {"arm": arm, "candidate_index": index, "root_sha256": self.root,
                    "task_id": self.task, "prefix_hex": prefix.tobytes().hex(),
                    "rgb_history": rgb, "action_history": history,
                    "capture_receipt_sha256": receipt, "terminated": False, "truncated": False}
        except BaseException as exc:
            self.failed = True
            self.write(stem + "_failure.json", canonical({"error": str(exc), "type": type(exc).__name__,
                "steps_attempted": self.steps_attempted, "steps_returned": self.steps_returned}))
            raise
        finally:
            if session is not None:
                session.close()

    def replay_winner(self, arm, winner):
        need(not self.failed and re.fullmatch(r"[a-z_]+", arm or ""), "replay route")
        index = winner["candidate_index"]
        need((arm, index) in self.receipts, "unknown capture")
        receipt, expected = self.receipts[(arm, index)]
        need(winner["feedback"]["capture_receipt_sha256"] == receipt, "winner receipt")
        plan = actions(np.frombuffer(bytes.fromhex(winner["raw_float32_hex"]), dtype="<f4").reshape(10, 2), 10)
        need(plan[:5].tobytes().hex() == expected["prefix_hex"], "winner prefix")
        stem = "replay_" + arm
        session = None
        try:
            self.write(stem + "_intent.json", canonical({"winner": winner, "capture_receipt_sha256": receipt}))
            session, root = self.start()
            last = self.advance(session, plan[:5])
            rgb = np.concatenate((np.asarray(root["rgb"]), np.asarray(last["rgb"])[None]))
            history = np.stack((self.history[1], normalization.normalized_history(plan[:5])[0])).astype("<f4")
            need([digest(x.tobytes()) for x in rgb] == expected["rgb_sha256"] and
                 history.tobytes().hex() == expected["action_history_hex"] and
                 root["guard"] == expected["guard"] and last["snapshot"] == expected["snapshot"],
                 "replay mismatch")
            self.write(stem + "_guard.json", canonical({"receipt": receipt, "accepted": True}))
            self.advance(session, plan[5:])
            self.write(stem + ".json", canonical({"status": "completed", "receipt": receipt,
                "steps_attempted": self.steps_attempted, "steps_returned": self.steps_returned,
                "unmetered": ["reconstruction_steps", "native_substeps", "render", "encoding",
                              "model", "wall_time", "flops"]}))
        except BaseException as exc:
            self.failed = True
            self.write(stem + "_failure.json", canonical({"error": str(exc), "type": type(exc).__name__,
                "steps_attempted": self.steps_attempted, "steps_returned": self.steps_returned}))
            raise
        finally:
            if session is not None:
                session.close()
