"""Opt-in durable driver for the P2 hook bridge; no model or physics launcher.

Each native event is fsynced by the existing exclusive hash-chained Journal.
Only callbacks explicitly passed to call() are guarded. A journal digest needs
an independent external anchor; it does not prove complete instrumentation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import threading

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_durable_hook_driver_20261030_codex"
MAX_JOURNAL = 32 * 1024 * 1024


def source_hashes():
    hashes = {name: spec.sha(Path(module.__file__).read_bytes()) for name, module in
              (("bridge", bridge), ("ledger", core), ("journal", transport), ("contract", spec))}
    hashes["driver"] = spec.sha(Path(__file__).read_bytes())
    return hashes


class Driver:
    """Explicit callback interface for a future new caller, not a sealed runner."""
    def __init__(self, directory, contract, contract_sha, original_caps, arm, run_id,
                 *, journal_limit=MAX_JOURNAL):
        caps = bridge.projected_caps(contract, contract_sha, original_caps)
        core.require(type(arm) is str and arm in contract["arms"], "contract arm")
        core.require(type(run_id) is str and run_id and run_id.strip() == run_id, "run id")
        core.require(type(journal_limit) is int and 0 < journal_limit <= MAX_JOURNAL, "journal bound")
        directory = Path(directory)
        core.require(directory.is_absolute() and not directory.exists() and
                     not directory.is_symlink() and directory.parent.is_dir(), "exclusive absolute directory")
        directory.mkdir()
        fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.directory = directory
        self.journal = transport.Journal(directory / "events.jsonl", journal_limit)
        self.closed = False
        self.failed = False
        self.depth = 0
        self.lock = threading.RLock()
        try:
            self.journal.emit("header", {"schema": NAME + ".header.v1", "contract": copy.deepcopy(contract),
                                         "contract_sha256": contract_sha, "arm": arm, "run_id": run_id,
                                         "original_caps": copy.deepcopy(original_caps), "projected_caps": caps,
                                         "source_sha256": source_hashes()})
            self.ledger = bridge.HookBridge(contract, contract_sha, original_caps,
                                            sink=lambda event: self.journal.emit("resource", event))
        except BaseException:
            self.closed = True
            self.journal.close()
            raise

    def call(self, scope, resource, fn, *, shape=None, directions=None):
        core.require(self.lock.acquire(blocking=False), "concurrent driver use forbidden")
        try:
            core.require(not self.closed and not self.failed, "driver is closed or fail-stopped")
            core.require(callable(fn), "callback required")
            old_scope = self.ledger.scope
            self.ledger.scope = scope
            self.depth += 1
            try:
                return self.ledger.call(resource, fn, shape=shape, directions=directions)
            except BaseException:
                self.failed = True
                raise
            finally:
                self.depth -= 1
                if old_scope is not None:
                    self.ledger.scope = old_scope
        finally:
            self.lock.release()

    def finish(self):
        core.require(self.lock.acquire(blocking=False), "concurrent driver use forbidden")
        try:
            core.require(not self.closed and self.depth == 0, "finish only outside active callbacks")
            try:
                receipt = self.ledger.receipt()
                core.audit_ledger(receipt)
                status = "failed_or_blocked" if self.failed else "complete"
                self.journal.emit("terminal", {"schema": NAME + ".terminal.v1",
                                               "status": status, "ledger": receipt})
            finally:
                self.closed = True
                self.journal.close()
            body = (self.directory / "events.jsonl").read_bytes()
            return {"journal": str(self.directory / "events.jsonl"), "sha256": hashlib.sha256(body).hexdigest(),
                    "status": status, "contract_receipt_ready": False, "P1_admission": False,
                    "P2_admission": False}
        finally:
            self.lock.release()


def read_chain(raw):
    """Validate the exact bytes already hashed, without reopening the path."""
    core.require(raw.endswith(b"\n"), "incomplete journal")
    rows, previous = [], None
    for line in raw.splitlines():
        event = json.loads(line, object_pairs_hook=spec.pairs_unique)
        core.require(set(event) == {"index", "previous", "kind", "payload", "sha256"}
                     and transport.canonical(event) == line, "canonical event")
        body = {k: event[k] for k in ("index", "previous", "kind", "payload")}
        core.require(type(event["index"]) is int and event["index"] == len(rows) and
                     event["previous"] == previous and event["sha256"] == transport.digest(body),
                     "journal chain")
        rows.append(event)
        previous = event["sha256"]
    return rows


def verify(path, expected_sha, expected_contract_sha):
    """Verify externally anchored full journal, replay native counts and lifecycle."""
    path = Path(path)
    core.require(spec.hash_string(expected_sha) and path.is_file() and not path.is_symlink()
                 and path.stat().st_size <= MAX_JOURNAL, "bounded journal and external digest required")
    raw = path.read_bytes()
    core.require(hashlib.sha256(raw).hexdigest() == expected_sha, "journal SHA mismatch")
    rows = read_chain(raw)
    core.require(len(rows) >= 2 and rows[0]["kind"] == "header" and rows[-1]["kind"] == "terminal"
                 and all(row["kind"] == "resource" for row in rows[1:-1]), "complete journal lifecycle")
    h, terminal = rows[0]["payload"], rows[-1]["payload"]
    core.require(set(h) == {"schema", "contract", "contract_sha256", "arm", "run_id",
                            "original_caps", "projected_caps", "source_sha256"}
                 and h["schema"] == NAME + ".header.v1", "header schema")
    core.require(spec.hash_string(expected_contract_sha) and h["contract_sha256"] == expected_contract_sha,
                 "external contract commitment mismatch")
    caps = bridge.projected_caps(h["contract"], h["contract_sha256"], h["original_caps"])
    core.require(caps == h["projected_caps"] and h["arm"] in h["contract"]["arms"] and
                 type(h["run_id"]) is str and h["run_id"] and h["run_id"].strip() == h["run_id"],
                 "contract/cap/run binding")
    core.require(h["source_sha256"] == source_hashes(), "source bytes drift")
    core.require(set(terminal) == {"schema", "status", "ledger"} and
                 terminal["schema"] == NAME + ".terminal.v1" and
                 terminal["status"] in ("complete", "failed_or_blocked"), "terminal schema")
    receipt = terminal["ledger"]
    core.require(receipt["caps"] == caps, "ledger projected caps")
    core.audit_ledger(receipt)
    expected = []
    for event in receipt["events"]:
        if event["status"] == "blocked_before_dispatch":
            expected.append({"kind": "resource_blocked", "entry": event})
        else:
            attempt = dict(event, status="attempted", ended_ns=None, error=None)
            expected.append({"kind": "resource_attempt", "entry": attempt})
            expected.append({"kind": "resource_return" if event["status"] == "returned"
                             else "resource_failure", "entry": event})
    actual = [r["payload"] for r in rows[1:-1]]
    core.require(sorted(map(core.canonical, actual)) == sorted(map(core.canonical, expected)),
                 "event/receipt bijection")
    stack = []
    failed_boundary = False
    for e in actual:
        if e["kind"] == "resource_attempt":
            core.require(not failed_boundary, "no dispatch after failure or refusal")
            stack.append(e["entry"]["index"])
        elif e["kind"] == "resource_blocked":
            failed_boundary = True
        else:
            core.require(stack and stack.pop() == e["entry"]["index"], "nested terminal ordering")
            if e["kind"] == "resource_failure":
                failed_boundary = True
    core.require(not stack, "no pending operations")
    failed = any(e["status"] != "returned" for e in receipt["events"])
    core.require(terminal["status"] == "failed_or_blocked" or not failed,
                 "failed events cannot have complete status")
    return {"schema": NAME + ".verification.v1", "journal_sha256": expected_sha,
            "status": terminal["status"], "attempted": receipt["attempted"],
            "completed": receipt["completed"], "events": len(receipt["events"]),
            "contract_receipt_ready": False, "coverage_verified": False,
            "full_vector_predispatch_guard": False, "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.journal, args.sha256, args.contract_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
