"""CPU-only regression and corruption tests for the saved-result auditor."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/"
          "codex_p1_two_root_saved_audit_20261101_exclusive.py")
spec = importlib.util.spec_from_file_location("exclusive_two_root_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_saved_two_roots():
    a, b = audit.audit(0), audit.audit(1)
    assert (a["steps"], b["steps"]) == (15, 6)
    assert (a["sealed_boundaries"], b["sealed_boundaries"]) == ([0, 5, 10], [0, 5])
    assert a["model_sha256"] == b["model_sha256"]
    assert a["root_sha256"] != b["root_sha256"]


@pytest.mark.parametrize("mode", ["physical", "cem", "ledger"])
def test_chain_rejects_tamper_and_truncation(tmp_path, mode):
    prev = "0" * 64 if mode == "cem" else None
    if mode == "physical":
        row = {"index": 0, "previous_sha256": prev, "kind": "test", "value": 1}
        h = hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()
        row["event_sha256"] = h
    else:
        row = {"index": 0, "previous": prev, "payload": {"value": 1}}
        row["sha256"] = audit.digest(row)
    path = tmp_path / "events.jsonl"
    original = json.dumps(row).encode() + b"\n"
    path.write_bytes(original)
    assert len(audit.chain(path, mode)) == 1
    path.write_bytes(original.replace(b'"value": 1', b'"value": 2'))
    with pytest.raises(ValueError, match="chain hash"):
        audit.chain(path, mode)
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        audit.chain(path, mode)


def test_native_success_uses_all_xy_and_angle_not_velocity():
    goal = [0.0] * 7
    assert audit.native([10, 0, 10, 0, 0.1, 1000, 1000], goal)[3]
    assert not audit.native([19, 0, 19, 0, 0, 0, 0], goal)[3]
    assert not audit.native([0, 0, 0, 0, math.pi / 9, 0, 0], goal)[3]
