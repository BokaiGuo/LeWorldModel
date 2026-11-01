"""Explicit, cooperative durable four-arm dispatch; no model imports or launcher."""
import copy
import hashlib
import os
from pathlib import Path
import threading

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec

NAME = "codex_four_arm_durable_20261030"
MAX_BYTES = 32 * 1024 * 1024


class Driver:
    def __init__(self, directory, contract, contract_sha, arm, level, run_id):
        spec.preflight(contract, contract_sha)
        spec.require(arm in spec.ARMS and type(run_id) is str and run_id and run_id.strip() == run_id,
                     "arm/run")
        levels = [v for v in contract["levels"] if v["id"] == level]
        spec.require(len(levels) == 1, "level")
        path = Path(directory)
        spec.require(path.is_absolute() and path.parent.is_dir() and not path.exists()
                     and not path.is_symlink(), "exclusive absolute directory")
        path.mkdir()
        self.path = path / "events.jsonl"
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.index, self.previous, self.size = 0, None, 0
        self.failed = self.closed = False
        self.lock = threading.Lock()
        self.caps = copy.deepcopy(levels[0]["caps"])
        self.charged = dict.fromkeys(spec.AXES, 0)
        try:
            for directory_path in (path, path.parent):
                fd = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self._emit("header", {"schema": NAME + ".v1", "contract": copy.deepcopy(contract),
                                  "contract_sha256": contract_sha, "arm": arm, "level": level,
                                  "run_id": run_id})
        except BaseException:
            os.close(self.fd)
            self.closed = True
            raise

    def _emit(self, kind, payload):
        body = {"index": self.index, "previous": self.previous, "kind": kind, "payload": payload}
        sha = spec.digest(body)
        raw = spec.canonical({**body, "sha256": sha}) + b"\n"
        spec.require(self.size + len(raw) <= MAX_BYTES, "journal size cap")
        view = memoryview(raw)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise OSError("journal short write")
            view = view[written:]
        os.fsync(self.fd)
        self.size += len(raw)
        self.previous, self.index = sha, self.index + 1

    def dispatch(self, phase, kind, reservation, fn):
        if not self.lock.acquire(blocking=False):
            self.failed = True
            raise ValueError("nested or concurrent dispatch forbidden")
        try:
            spec.require(not self.closed and not self.failed, "driver inactive")
            try:
                spec.require(phase in spec.PHASES and kind in spec.KINDS and callable(fn), "phase/kind/callback")
                spec.vector(reservation)
                spec.require(reservation[spec.KINDS[kind]] > 0, "operation charge required")
                spec.require((kind != "encode" or reservation["encode_samples"] > 0)
                             and (kind != "predict" or reservation["predict_samples"] > 0),
                             "sample charge required")
            except BaseException:
                self.failed = True
                self._emit("invalid", {})
                raise
            cost = dict(reservation)
            payload = {"phase": phase, "kind": kind, "reservation": cost}
            if any(self.charged[k] + cost[k] > self.caps[k] for k in spec.AXES):
                self.failed = True
                self._emit("blocked", payload)
                raise ValueError("budget exceeded before dispatch")
            self._emit("intent", payload)
            for axis in spec.AXES:
                self.charged[axis] += cost[axis]
            try:
                result = fn()
                spec.require(not self.failed, "nested or concurrent dispatch attempted")
            except BaseException:
                self.failed = True
                self._emit("failure", {})
                raise
            self._emit("return", {})
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def finish(self):
        spec.require(self.lock.acquire(blocking=False), "driver busy")
        try:
            spec.require(not self.closed, "driver closed")
            status = "failed_or_blocked" if self.failed else "closed_unverified"
            try:
                self._emit("terminal", {"status": status, "charged_upper_bounds": self.charged})
            finally:
                self.closed = True
                os.close(self.fd)
            raw = self.path.read_bytes()
            return {"journal": str(self.path), "sha256": hashlib.sha256(raw).hexdigest(), "status": status}
        finally:
            self.lock.release()
