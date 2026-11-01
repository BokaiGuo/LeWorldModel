import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_reset_ledger_recovery_map_20261030 as m


@pytest.fixture
def inputs():
    log = m.field(m.pinned(m.INDEX), "original_dispatch_log", 32768)
    sources = m.field(m.pinned(m.COVERAGE), "sources", 32768)
    receipts = m.delta_receipts(m.pinned(m.DELTA))
    return log, receipts, sources


def test_real_process_chain_and_scope(inputs):
    log, receipts, sources = inputs
    rows = m.merge(log, receipts, sources)
    assert len(rows) == 10
    assert sum(x["source_id"] == "B" for x in rows) == 3
    assert sum(x["process_status"] == "exit_recorded" for x in rows) == 3
    assert sum(x["process_status"] == "traceback_recorded" for x in rows) == 2
    assert sum(x["process_status"] == "stdout_without_exit" for x in rows) == 5
    assert {r["source_id"]: r["return_receipts"][0]["session_id"]
            for r in rows if r["source_id"] in "AEH"} == {"A": 37004, "E": 29686, "H": 59270}
    assert all(r["reset_attempt_ledger"] == "not_established" for r in rows)
    assert m.verify_original_lines(m.LOG, log, rows) == 23


@pytest.mark.parametrize("mutation", ["missing_call", "duplicate_dispatch", "orphan",
                                      "session", "source", "wrong_log", "missing_b"])
def test_fail_closed(inputs, mutation):
    log, receipts, sources = copy.deepcopy(inputs)
    if mutation == "missing_call":
        receipts.pop()
    elif mutation == "duplicate_dispatch":
        log["dispatches"][0]["call_id"] = log["dispatches"][1]["call_id"]
    elif mutation == "orphan":
        receipts[0]["dispatch_call_id"] = "other"
    elif mutation == "session":
        next(r for r in receipts if r["operation"] == "write_stdin")["polled_session"] = 1
    elif mutation == "source":
        receipts[0]["source_id"] = "A"
    elif mutation == "wrong_log":
        log["sha256"] = "0" * 64
    else:
        log["dispatches"][0]["source_id"] = "A"
        receipts[0]["source_id"] = "A"
    with pytest.raises(ValueError):
        m.merge(log, receipts, sources)


def test_tampered_delta_rejected():
    envelope = json.loads(m.pinned(m.DELTA))
    envelope["semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.delta_receipts(json.dumps(envelope).encode())


def test_original_line_mutation_rejected(inputs, tmp_path, monkeypatch):
    line = b"first\n"
    path = tmp_path / "log"
    path.write_bytes(line)
    monkeypatch.setattr(m, "LOG_BYTES", len(line))
    monkeypatch.setattr(m, "LOG_SHA", hashlib.sha256(line).hexdigest())
    log = {"dispatches": [{"line": 1, "line_sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="line hash"):
        m.verify_original_lines(path, log, [])


def test_field_projection_only():
    raw = b'{"outcome":{"private":"must not be returned"},"sources":[{"id":"A"}]}'
    assert m.field(raw, "sources") == [{"id": "A"}]
