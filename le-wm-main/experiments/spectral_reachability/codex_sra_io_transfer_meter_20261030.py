"""Explicit-operation resource meter; no implicit GPU, loader or simulator.

File reads count bytes returned to this wrapper, tensor copies count logical
payload bytes, and model loads count callback entries. The ledger is a local
single-process prototype; uninstrumented work and physical I/O remain unknown.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

AXES = ("file_read_bytes", "host_to_device_payload_bytes", "device_to_host_payload_bytes",
        "model_load_attempts", "load_wall_ns")
UNKNOWN = {"total_file_io_bytes": None, "physical_disk_io_bytes": None,
           "total_device_transfer_bytes": None, "physical_bus_bytes": None,
           "model_load_internal_io_bytes": None, "model_load_internal_transfer_bytes": None,
           "total_compute_flops": None, "equal_total_budget": False}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Ledger:
    """Reserve before callbacks; failure keeps pending reservations, no resume."""
    def __init__(self, path, *, binding_sha256, caps, source_sha256):
        need(valid_sha(binding_sha256) and valid_sha(source_sha256), "external SHA bindings required")
        need(isinstance(caps, dict) and set(caps) == set(AXES) and
             all(value is None or type(value) is int and value >= 0 for value in caps.values()),
             "explicit per-axis cap or unknown required")
        need(source_sha256 == sha(Path(__file__).read_bytes()), "meter source mismatch")
        self.path = Path(path)
        self.caps = dict(caps)
        self.attempted = {a: 0 for a in AXES}
        self.completed = {a: 0 for a in AXES}
        self.pending, self.seen = {}, set()
        self.tail, self.seq, self.closed, self.poisoned = "0" * 64, 0, False, False
        self.file = self.path.open("xb")
        try:
            sync_dir(self.path.parent)
            self._append("header", {"schema": "codex_sra_explicit_resource_ledger_v1",
                                    "binding_sha256": binding_sha256, "source_sha256": source_sha256,
                                    "caps": self.caps, "coverage": "explicit_operations_only"})
        except BaseException:
            self.poisoned = True
            self.file.close()
            raise

    def _append(self, kind, payload):
        need(not self.closed and not self.poisoned, "ledger unavailable")
        body = {"seq": self.seq, "prev": self.tail, "kind": kind, "payload": payload}
        checksum = sha(canonical(body))
        try:
            self.file.write(canonical({**body, "sha256": checksum}) + b"\n")
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException:
            self.poisoned = True
            raise
        self.tail, self.seq = checksum, self.seq + 1

    def _start(self, op_id, kind, amounts, details):
        need(isinstance(op_id, str) and op_id and op_id not in self.seen, "operation id")
        need(isinstance(amounts, dict) and amounts and set(amounts) <= set(AXES) and
             all(type(v) is int and v > 0 for v in amounts.values()), "positive reservations required")
        need(all(self.caps[a] is not None and self.attempted[a] + v <= self.caps[a]
                 for a, v in amounts.items()), "missing or exhausted resource cap")
        check_attempt(kind, amounts, details)
        self._append("attempt", {"id": op_id, "kind": kind, "reserved": amounts, "details": details})
        for a, value in amounts.items():
            self.attempted[a] += value
        self.seen.add(op_id)
        self.pending[op_id] = {"kind": kind, "reserved": dict(amounts), "details": details}

    def _returned(self, op_id, observed):
        pending = self.pending[op_id]
        self._append("return", {"id": op_id, "observed": observed})
        for a, value in pending["reserved"].items():
            self.completed[a] += value
        del self.pending[op_id]

    def _failed(self, op_id, exc):
        if not self.poisoned:
            self._append("failure", {"id": op_id, "error_type": type(exc).__name__})

    def read_asset(self, op_id, path, *, expected_sha256):
        """Read one regular file through this interface, recording returned bytes."""
        need(valid_sha(expected_sha256), "expected asset hash")
        with Path(path).open("rb") as f:
            before = os.fstat(f.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_size > 0, "nonempty regular asset required")
            amount = before.st_size
            self._start(op_id, "file_read", {"file_read_bytes": amount},
                        {"path": str(Path(path).resolve()), "expected_sha256": expected_sha256})
            try:
                raw = f.read(amount)
                after = os.fstat(f.fileno())
                need(len(raw) == amount and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                     (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "short or changed asset")
                need(sha(raw) == expected_sha256, "asset SHA mismatch")
                self._returned(op_id, {"logical_read_bytes": len(raw), "content_sha256": sha(raw)})
                return raw
            except BaseException as exc:
                self._failed(op_id, exc)
                raise

    def transfer(self, op_id, tensor, destination, *, copy_fn=None):
        """Count a tensor's logical cross-device payload, not physical bus bytes."""
        src = str(tensor.device.type)
        dst = str(destination.type)
        need((src, dst) in (("cpu", "cuda"), ("cuda", "cpu")), "CPU/CUDA direction required")
        axis = "host_to_device_payload_bytes" if src == "cpu" else "device_to_host_payload_bytes"
        size = int(tensor.numel()) * int(tensor.element_size())
        need(size > 0, "positive tensor payload required")
        self._start(op_id, "tensor_transfer", {axis: size},
                    {"source": src, "destination": dst, "payload_bytes": size,
                     "fixture": copy_fn is not None})
        try:
            result = (copy_fn(tensor, destination) if copy_fn is not None else
                      tensor.to(destination, non_blocking=False))
            need(str(result.device.type) == dst and int(result.numel()) * int(result.element_size()) == size,
                 "returned tensor direction or size")
            self._returned(op_id, {"logical_payload_bytes": size, "hardware_transfer_verified": False})
            return result
        except BaseException as exc:
            self._failed(op_id, exc)
            raise

    def load_model(self, op_id, callback, *, wall_limit_ns):
        """Count one loader callback and elapsed time; internal I/O remains unknown."""
        need(callable(callback) and type(wall_limit_ns) is int and wall_limit_ns > 0, "load reservation")
        self._start(op_id, "model_load", {"model_load_attempts": 1, "load_wall_ns": wall_limit_ns},
                    {"wall_limit_is_preemptive": False})
        started = time.monotonic_ns()
        try:
            model = callback()
            elapsed = time.monotonic_ns() - started
            need(model is not None, "loader returned no model")
            need(elapsed <= wall_limit_ns, "model load wall reservation exceeded")
            self._returned(op_id, {"elapsed_ns": elapsed, "internal_io_bytes": None,
                                   "internal_transfer_bytes": None})
            return model
        except BaseException as exc:
            self._failed(op_id, exc)
            raise

    def finish(self):
        need(not self.closed and not self.poisoned, "ledger unavailable")
        snapshot = {"attempted_reservations": self.attempted, "returned_reservations": self.completed,
                    "pending": self.pending, "coverage": "explicit_operations_only", **UNKNOWN}
        self._append("footer", snapshot)
        self.file.close()
        self.closed = True
        return sha(self.path.read_bytes())

    def close_partial(self):
        if not self.closed:
            self.file.close()
            self.closed = True


def consume(path, expected_sha256, binding_sha256, source_sha256):
    raw = Path(path).read_bytes()
    need(valid_sha(expected_sha256) and sha(raw) == expected_sha256 and raw.endswith(b"\n"), "ledger anchor")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    events = [json.loads(line, object_pairs_hook=unique) for line in raw.splitlines()]
    tail = "0" * 64
    for i, event in enumerate(events):
        need(set(event) == {"seq", "prev", "kind", "payload", "sha256"} and type(event["seq"]) is int and event["seq"] == i
             and event["prev"] == tail, "ledger sequence")
        body = {key: event[key] for key in ("seq", "prev", "kind", "payload")}
        need(sha(canonical(body)) == event["sha256"], "ledger chain")
        tail = event["sha256"]
    need(len(events) >= 2 and events[0]["kind"] == "header" and events[-1]["kind"] == "footer",
         "ledger footer")
    header = events[0]["payload"]
    need(header["schema"] == "codex_sra_explicit_resource_ledger_v1"
         and header["binding_sha256"] == binding_sha256 and header["source_sha256"] == source_sha256
         and source_sha256 == sha(Path(__file__).read_bytes())
         and header["coverage"] == "explicit_operations_only", "ledger identity")
    caps = header["caps"]
    need(set(caps) == set(AXES) and all(v is None or type(v) is int and v >= 0 for v in caps.values()), "caps")
    attempted = {a: 0 for a in AXES}
    completed = {a: 0 for a in AXES}
    pending, seen, observed, failed = {}, set(), [], set()
    for event in events[1:-1]:
        kind, payload = event["kind"], event["payload"]
        op_id = payload["id"]
        if kind == "attempt":
            reserved = payload["reserved"]
            need(op_id not in seen and payload["kind"] in ("file_read", "tensor_transfer", "model_load")
                 and reserved and set(reserved) <= set(AXES)
                 and all(type(v) is int and v > 0 and caps[a] is not None and attempted[a] + v <= caps[a]
                         for a, v in reserved.items()), "attempt/cap")
            check_attempt(payload["kind"], reserved, payload["details"])
            seen.add(op_id)
            for a, v in reserved.items():
                attempted[a] += v
            pending[op_id] = {"kind": payload["kind"], "reserved": reserved, "details": payload["details"]}
        elif kind == "return":
            need(op_id in pending and op_id not in failed, "return without live attempt")
            entry = pending.pop(op_id)
            check_observation(entry, payload["observed"])
            for a, v in entry["reserved"].items():
                completed[a] += v
            observed.append({"id": op_id, "kind": entry["kind"], "details": entry["details"],
                             "observed": payload["observed"]})
        elif kind == "failure":
            need(op_id in pending and op_id not in failed and isinstance(payload["error_type"], str), "failure")
            failed.add(op_id)
        else:
            raise ValueError("unknown ledger event")
    expected = {"attempted_reservations": attempted, "returned_reservations": completed,
                "pending": pending, "coverage": "explicit_operations_only", **UNKNOWN}
    need(events[-1]["payload"] == expected, "footer mismatch")
    return {"snapshot": expected, "observed": observed, "ledger_sha256": expected_sha256,
            "callbacks_reexecuted": False, "new_gpu_or_physical_calls": 0}


def check_attempt(kind, amounts, details):
    need(isinstance(details, dict), "operation details")
    if kind == "file_read":
        need(set(amounts) == {"file_read_bytes"} and set(details) == {"path", "expected_sha256"}
             and isinstance(details["path"], str) and details["path"] and valid_sha(details["expected_sha256"]),
             "file attempt")
    elif kind == "tensor_transfer":
        need(set(details) == {"source", "destination", "payload_bytes", "fixture"} and
             (details["source"], details["destination"]) in (("cpu", "cuda"), ("cuda", "cpu"))
             and type(details["fixture"]) is bool and type(details["payload_bytes"]) is int
             and details["payload_bytes"] > 0, "transfer attempt")
        axis = ("host_to_device_payload_bytes" if details["source"] == "cpu"
                else "device_to_host_payload_bytes")
        need(amounts == {axis: details["payload_bytes"]}, "transfer charge")
    elif kind == "model_load":
        need(set(amounts) == {"model_load_attempts", "load_wall_ns"} and
             amounts["model_load_attempts"] == 1 and details == {"wall_limit_is_preemptive": False},
             "load attempt")
    else:
        raise ValueError("operation kind")


def check_observation(entry, observed):
    kind, amounts, details = entry["kind"], entry["reserved"], entry["details"]
    if kind == "file_read":
        need(set(amounts) == {"file_read_bytes"} and
             observed == {"logical_read_bytes": amounts["file_read_bytes"],
                          "content_sha256": details["expected_sha256"]}, "file observation")
    elif kind == "tensor_transfer":
        direction = (details["source"], details["destination"])
        need(direction in (("cpu", "cuda"), ("cuda", "cpu")) and type(details["fixture"]) is bool, "direction")
        axis = "host_to_device_payload_bytes" if direction == ("cpu", "cuda") else "device_to_host_payload_bytes"
        need(amounts == {axis: details["payload_bytes"]} and
             observed == {"logical_payload_bytes": details["payload_bytes"], "hardware_transfer_verified": False},
             "transfer observation")
    else:
        need(set(amounts) == {"model_load_attempts", "load_wall_ns"} and amounts["model_load_attempts"] == 1
             and details == {"wall_limit_is_preemptive": False} and
             set(observed) == {"elapsed_ns", "internal_io_bytes", "internal_transfer_bytes"} and
             type(observed["elapsed_ns"]) is int and 0 <= observed["elapsed_ns"] <= amounts["load_wall_ns"]
             and observed["internal_io_bytes"] is None and observed["internal_transfer_bytes"] is None,
             "model load observation")
