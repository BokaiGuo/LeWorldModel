"""Exclusive fsynced event journal and saved-only replay for the paired RAM interface.

This module has no model/environment imports and does not dispatch physical work.
Incomplete transactions are deliberately not resumable or consumable.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve()
BASE = HERE.with_name("codex_paired_physical_interface_20261030.py")
BASE_SHA = "8efd38b1fac3fdba00d6ea3324a404db921f6d310ddd1b0c0a2e6cbce8492766"
MAX_BYTES = 16 * 1024 * 1024
OPS = {"observe", "record_resource_attempt", "record_resource_return", "select",
       "begin_step", "finish_step", "stop"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hashbytes(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_base():
    require(hashbytes(BASE.read_bytes()) == BASE_SHA, "paired interface source drift")
    spec = importlib.util.spec_from_file_location("codex_paired_pinned_interface", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_protocol(config, persist):
    require(set(config) == {"identities", "initial_frames", "initial_actions", "horizon", "packet_steps", "caps"},
            "unexpected protocol configuration")
    m = load_base()
    return m.PairedProtocol(**copy.deepcopy(config), persist=persist)


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Journal:
    def __init__(self, path, config):
        self.path = Path(path)
        self.config = json.loads(canonical(config))
        self.protocol = make_protocol(self.config, self._seal)
        self.file = self.path.open("xb")
        self.seq, self.tail, self.size, self.poisoned, self.closed = 0, "0" * 64, 0, False, False
        try:
            sync_dir(self.path.parent)
            self._append("header", {"schema": "codex_paired_journal_20261030_v1",
                                    "base_sha256": BASE_SHA, "journal_source_sha256": hashbytes(HERE.read_bytes()),
                                    "config": self.config})
        except BaseException:
            self.poisoned = True
            self.file.close()
            raise

    def _append(self, kind, payload):
        require(not self.poisoned and not self.closed, "journal closed or poisoned")
        body = {"seq": self.seq, "prev": self.tail, "kind": kind, "payload": payload}
        checksum = hashbytes(canonical(body))
        raw = canonical({**body, "sha256": checksum}) + b"\n"
        require(self.size + len(raw) <= MAX_BYTES, "journal byte cap")
        try:
            self.file.write(raw)
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException:
            self.poisoned = True
            raise
        self.seq += 1
        self.tail = checksum
        self.size += len(raw)

    def _seal(self, value):
        sha = hashbytes(canonical(value))
        self._append("seal", {"value": value, "content_sha256": sha})
        return sha

    def call(self, operation, *args, **kwargs):
        require(not self.poisoned and not self.closed, "journal closed or poisoned")
        require(operation in OPS, "operation forbidden")
        arguments = json.loads(canonical({"args": args, "kwargs": kwargs}))
        try:
            self._append("begin", {"operation": operation, **arguments})
            result = getattr(self.protocol, operation)(*arguments["args"], **arguments["kwargs"])
            self._append("commit", {"result": result, "snapshot_sha256": hashbytes(canonical(self.protocol.snapshot()))})
            return copy.deepcopy(result)
        except BaseException:
            # An incomplete transaction is evidence, not permission to replay dispatch.
            self.poisoned = True
            raise

    def finish(self):
        require(not self.poisoned and not self.closed, "journal closed or poisoned")
        snap = self.protocol.snapshot()
        require(all(r["status"] in ("complete", "stopped") for r in snap["rows"]), "all fixed arms must be terminal")
        self._append("footer", {"snapshot": snap})
        self.file.close()
        self.closed = True
        return hashbytes(self.path.read_bytes())

    def close_partial(self):
        if not self.closed:
            self.file.close()
            self.closed = True


def no_duplicates(items):
    out = {}
    for key, value in items:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def consume(path, expected_sha256):
    """Verify byte anchor and replay every completed operation without dispatch."""
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64, "external byte SHA required")
    raw = Path(path).read_bytes()
    require(len(raw) <= MAX_BYTES and hashbytes(raw) == expected_sha256, "journal byte hash mismatch or too large")
    require(raw.endswith(b"\n"), "truncated journal")
    records = []
    tail = "0" * 64
    for n, line in enumerate(raw.splitlines()):
        item = json.loads(line, object_pairs_hook=no_duplicates,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        require(set(item) == {"seq", "prev", "kind", "payload", "sha256"} and
                type(item["seq"]) is int and item["seq"] == n and item["prev"] == tail, "event chain")
        body = {k: item[k] for k in ("seq", "prev", "kind", "payload")}
        require(hashbytes(canonical(body)) == item["sha256"], "event hash")
        tail = item["sha256"]
        records.append(item)
    require(len(records) >= 2 and records[0]["kind"] == "header" and records[-1]["kind"] == "footer",
            "missing header or terminal footer")
    header = records[0]["payload"]
    require(set(header) == {"schema", "base_sha256", "journal_source_sha256", "config"}
            and header["schema"] == "codex_paired_journal_20261030_v1"
            and header["base_sha256"] == BASE_SHA and header["journal_source_sha256"] == hashbytes(HERE.read_bytes()),
            "source binding mismatch")
    seals = []

    def check_seal(value):
        require(bool(seals), "unrecorded seal")
        stored = seals.pop(0)
        require(stored == {"value": value, "content_sha256": hashbytes(canonical(value))}, "seal mismatch")
        return stored["content_sha256"]

    machine = make_protocol(header["config"], check_seal)
    pos = 1
    transactions = 0
    while pos < len(records) - 1:
        r = records[pos]
        require(r["kind"] == "begin", "expected begin")
        begin = r["payload"]
        require(set(begin) == {"operation", "args", "kwargs"} and begin["operation"] in OPS
                and isinstance(begin["args"], list) and isinstance(begin["kwargs"], dict), "invalid transaction")
        pos += 1
        seals.clear()
        while pos < len(records) - 1 and records[pos]["kind"] == "seal":
            seals.append(records[pos]["payload"])
            pos += 1
        require(pos < len(records) - 1 and records[pos]["kind"] == "commit", "incomplete transaction")
        commit = records[pos]["payload"]
        require(set(commit) == {"result", "snapshot_sha256"}, "invalid commit")
        result = getattr(machine, begin["operation"])(*begin["args"], **begin["kwargs"])
        require(not seals and result == commit["result"]
                and hashbytes(canonical(machine.snapshot())) == commit["snapshot_sha256"], "replay mismatch")
        transactions += 1
        pos += 1
    snap = machine.snapshot()
    require(records[-1]["payload"] == {"snapshot": snap} and
            all(r["status"] in ("complete", "stopped") for r in snap["rows"]), "footer mismatch")
    return {"schema": "codex_paired_saved_consumer_20261030_v1",
            "journal_sha256": expected_sha256, "journal_chain_tail": tail,
            "transactions_replayed": transactions, "snapshot": snap,
            "new_neural_calls": 0, "new_physical_steps": 0,
            "limitations": ["hash anchor is not independent proof of execution or identity",
                            "journal is non-dispatching; no real runtime or upstream source closure verified",
                            "FLOPs and total compute equality remain unverified"]}


def main():
    parser = argparse.ArgumentParser(description="Read and replay a finalized paired journal; never dispatch.")
    parser.add_argument("--journal", required=True)
    parser.add_argument("--sha256", required=True, help="externally recorded byte SHA256")
    parser.add_argument("--output", required=True, help="new exclusive saved-consumer JSON")
    args = parser.parse_args()
    result = consume(args.journal, args.sha256)
    output = Path(args.output)
    with output.open("xb") as f:
        f.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        f.flush()
        os.fsync(f.fileno())
    sync_dir(output.parent)
    print("saved", output)


if __name__ == "__main__":
    main()
