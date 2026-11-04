"""Capture-to-consumer development bridge. Does not authorize physical execution.

Session factory, provenance, environment lifecycle, lock and budgets remain
the caller's responsibility. Journal is a test store, not a native Store.
"""
import json
import os
from pathlib import Path

import numpy as np

from . import p2_feedback_saved_consumer_20261102_codexexclusive as consumer
from . import p2_prefix_feedback_common_budget_20261102_codexexclusive as kernel


def need(value, message):
    if not value:
        raise ValueError(message)


def durable_append(path, data):
    with path.open("ab") as out:
        need(out.write(data) == len(data), "short journal write")
        out.flush()
        os.fsync(out.fileno())


class Journal:
    """Minimal durable fake-test store; no historical prefix authentication."""
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("xb") as f:
            f.flush()
            os.fsync(f.fileno())
        self.prefix_events = {0: []}

    def emit(self, kind, **fields):
        durable_append(self.path, kernel.canonical({"kind": kind, **fields}) + b"\n")

    def close(self):
        pass


class Archive:
    """Collect actual callback outputs as they occur, never authorize replay."""
    def __init__(self, path, factory, root, task, base_history, base_rgb, warmup, shared, budget):
        self.path = Path(path)
        need(self.path.is_dir() and not self.path.is_symlink() and callable(factory), "archive path/factory")
        need(isinstance(root, str) and len(root) == 64 and isinstance(task, str) and task, "root/task")
        self.factory, self.root, self.task, self.budget = factory, root, task, budget
        self.base_history = np.asarray(base_history, dtype="<f4").copy()
        self.base_rgb = list(base_rgb)
        self.warmup = np.asarray(warmup, dtype="<f4").copy()
        self.shared = np.asarray(shared, dtype="<f4").copy()
        need(self.base_history.shape == (2, 10) and np.isfinite(self.base_history).all() and
             self.warmup.shape == (10, 2) and self.shared.shape == (15, 2) and
             np.isfinite(self.warmup).all() and np.isfinite(self.shared).all() and
             len(self.base_rgb) == 2, "base inputs")
        self.captures, self.pending = {}, None
        self.failed = False
        self.event_path = self.path / "model_events.jsonl"
        with self.event_path.open("xb") as out:
            out.flush()
            os.fsync(out.fileno())

    def put(self, name, data):
        need(Path(name).name == name and isinstance(data, bytes), "artifact name/data")
        target = self.path / name
        need(not target.is_symlink(), "artifact symlink")
        with target.open("xb") as out:
            need(out.write(data) == len(data), "short artifact write")
            out.flush()
            os.fsync(out.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return self.ref(name)

    def ref(self, name):
        target = self.path / name
        need(Path(name).name == name and target.is_file() and not target.is_symlink(), "artifact")
        return {"path": name, "sha256": kernel.sha(target.read_bytes())}

    def emit(self, event):
        try:
            durable_append(self.event_path, kernel.canonical(event) + b"\n")
        except BaseException:
            self.failed = True
            raise

    def capture(self, arm, index, prefix):
        need(not self.failed and self.pending is None, "pending encode or failed archive")
        expected = (kernel.ARMS[len(self.captures) % 2], len(self.captures) // 2)
        need((arm, index) == expected and index < kernel.N, "fixed capture order")
        prefix = np.asarray(prefix)
        need(prefix.dtype == np.dtype("<f4") and prefix.shape == (5, 2) and
             np.isfinite(prefix).all() and (np.abs(prefix) <= 1).all(), "prefix bytes")
        name = f"{index:02d}_{arm}"
        session, store = None, None
        try:
            self.put(name + "_intent.json", kernel.canonical({"arm": arm, "candidate_index": index,
                "root_sha256": self.root, "task_id": self.task, "prefix_hex": prefix.tobytes().hex()}))
            session, store = self.factory(arm, index)
            root = session.reconstruct()
            base_pixels, initial = np.asarray(root["rgb"]), np.asarray(root["history"])
            need(root["root_sha256"] == self.root and root["task_id"] == self.task and
                 base_pixels.shape == (2, 224, 224, 3) and base_pixels.dtype == np.uint8 and
                 [kernel.sha(x.tobytes()) for x in base_pixels] == self.base_rgb and
                 initial.dtype == np.dtype("<f4") and initial.shape == (2, 10) and
                 initial.tobytes() == self.base_history.tobytes(), "reconstructed root/context")
            for command in prefix:
                last = session.step(command.copy())
                need(type(last["terminated"]) is bool and type(last["truncated"]) is bool,
                     "explicit stop flags")
                if last["terminated"] or last["truncated"]:
                    raise RuntimeError("native stop without padding")
            pixels = np.asarray(last["rgb"])
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8, "cp5 RGB")
            rgb = np.concatenate((base_pixels, pixels[None]))
            history = np.stack((self.base_history[1], kernel.normalization.normalized_history(prefix)[0])).astype("<f4")
            receipt = {"arm": arm, "candidate_index": index, "root_sha256": self.root, "task_id": self.task,
                       "prefix_hex": prefix.tobytes().hex(), "rgb_sha256": [kernel.sha(x.tobytes()) for x in rgb],
                       "action_history_hex": history.tobytes().hex(), "guard": root["guard"],
                       "snapshot": last["snapshot"], "terminated": last["terminated"], "truncated": last["truncated"]}
            need(receipt["terminated"] is False and receipt["truncated"] is False, "capture stopped")
            session.close()
            session = None
            if hasattr(store, "close"):
                store.close()
            journal_path = Path(store.path)
            if journal_path.is_dir() and not journal_path.is_symlink():
                need(journal_path.parent == self.path, "exclusive native store")
                journal_path = journal_path / "journal.jsonl"
            else:
                need(journal_path.parent == self.path and journal_path.name == name + "_native.jsonl",
                     "exclusive native journal")
            need(journal_path.is_file() and not journal_path.is_symlink(), "native journal")
            journal = journal_path.read_bytes()
            consumer.audit_native_journal(journal, {"warmup_hex": self.warmup.tobytes().hex(),
                "shared_prefix_hex": self.shared.tobytes().hex(), "root_sha256": self.root, "task_id": self.task},
                prefix.tobytes(), receipt)
            journal_ref = (self.ref(journal_path.name) if journal_path.parent == self.path else
                           self.put(name + "_native.jsonl", journal))
            refs = {"native_journal": journal_ref,
                    "rgb": self.put(name + "_rgb.bin", rgb.tobytes()),
                    "receipt": self.put(name + "_receipt.json", kernel.canonical(receipt))}
            self.captures[name] = refs
            self.pending = (name, rgb.copy(), history.copy())
            return {"arm": arm, "candidate_index": index, "root_sha256": self.root, "task_id": self.task,
                    "prefix_hex": prefix.tobytes().hex(), "rgb_history": rgb, "action_history": history,
                    "capture_receipt_sha256": refs["receipt"]["sha256"],
                    "terminated": receipt["terminated"], "truncated": receipt["truncated"]}
        except BaseException as exc:
            self.failed = True
            self.put(name + "_failure.json", kernel.canonical({"error_type": type(exc).__name__,
                "message": str(exc)[:500], "capture_complete": False}))
            raise
        finally:
            if session is not None:
                session.close()
            if session is not None and store is not None and hasattr(store, "close"):
                store.close()

    def encode(self, fn, rgb, history):
        need(not self.failed and self.pending is not None and callable(fn), "pending capture")
        name, expected_rgb, expected_history = self.pending
        try:
            need(np.asarray(rgb).dtype == np.uint8 and np.array_equal(rgb, expected_rgb) and
                 np.asarray(history).dtype == np.dtype("<f4") and
                 np.asarray(history).tobytes() == expected_history.tobytes(), "encoder input binding")
            context = fn(expected_rgb.copy(), expected_history.copy())
            kernel.validate_context(context, expected_history)
            self.captures[name]["context"] = self.put(name + "_context.json", kernel.canonical(context))
            self.pending = None
            return context
        except BaseException:
            self.failed = True
            raise

    def seal(self, bank, result, sources):
        need(not self.failed and self.pending is None and len(self.captures) == 32 and
             len(bank) == 16 and set(sources) == set(consumer.SOURCE_ROLES), "incomplete archive")
        need(self.budget.path == self.path / "budget.jsonl", "budget path")
        self.budget.file.flush()
        os.fsync(self.budget.file.fileno())
        events = [consumer.decode(line) for line in self.event_path.read_bytes().splitlines()]
        manifest = {"schema": consumer.SCHEMA, "root_sha256": self.root, "task_id": self.task,
                    "base_history": self.base_history.tolist(), "base_rgb_sha256": self.base_rgb,
                    "warmup_hex": self.warmup.tobytes().hex(), "shared_prefix_hex": self.shared.tobytes().hex(),
                    "sources": sources, "bank": self.put("bank.bin", b"".join(bank)),
                    "result": self.put("result.json", kernel.canonical(result)),
                    "events": self.put("events.json", kernel.canonical(events)), "captures": self.captures,
                    "budget": {"file": self.ref("budget.jsonl"), "tail_sha256": self.budget.previous,
                               "expected_counts": dict(self.budget.attempted)}}
        anchor = self.put("manifest.json", kernel.canonical(manifest))["sha256"]
        receipt = consumer.consume(self.path, anchor)
        need(receipt["physical_authorization"] is False and receipt["source_semantics_authenticated"] is False,
             "authority boundary")
        return anchor
