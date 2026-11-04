import hashlib
import json

import pytest

from experiments.spectral_reachability import d2p_root_state_source_binding_20261102_exclusive as audit


def test_real_source_and_all_indexed_match_bytes():
    index = audit.bounded.read_json(audit.bounded.INDEX, audit.bounded.INDEX_SHA)
    ref = index["observed_prefix_records"][0]["evidence"]
    result = audit.audit(ref["path"], ref["sha256"], ref["pointer"])
    assert result["candidate_source"]["derived_state_sha256"] == index["observed_prefix_records"][0]["root_state_sha256"]
    assert result["historical_match_count"] == 10
    assert not result["freshness_certified"]
    assert not result["training_disjoint_certified"]


def test_no_match_remains_fail_closed(tmp_path):
    path = tmp_path / "candidate.json"
    raw = json.dumps({"state": [1, 2, 3, 4, 5, 6, 7]}).encode()
    path.write_bytes(raw)
    result = audit.audit(path, hashlib.sha256(raw).hexdigest(), "/state")
    assert result["historical_match_count"] == 0
    assert result["verdict"] == "FAIL_CLOSED"
    assert not result["independent_confirmation_authorized"]


@pytest.mark.parametrize("state", [[1, 2], [1, 2, 3, 4, 5, 6, True],
                                  [1, 2, 3, 4, 5, 6, float("nan")]])
def test_bad_state_rejected(state):
    with pytest.raises(ValueError):
        audit.state_digest(state)


def test_byte_drift_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"state":[1,2,3,4,5,6,7]}')
    with pytest.raises(ValueError):
        audit.read_source(path, "0" * 64)


def test_pointer_validation():
    assert audit.pointer({"a/b": [1]}, "/a~1b/0") == 1
    with pytest.raises(ValueError):
        audit.pointer({"a": [1]}, "/a/00")
    with pytest.raises(ValueError):
        audit.pointer({}, "/~2")


def test_symlink_source_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"state":[1,2,3,4,5,6,7]}')
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        audit.read_source(link, hashlib.sha256(path.read_bytes()).hexdigest())


def test_matched_historical_source_drift_blocks(monkeypatch):
    index = audit.bounded.read_json(audit.bounded.INDEX, audit.bounded.INDEX_SHA)
    ref = index["observed_prefix_records"][0]["evidence"]
    original = audit.read_source
    calls = []

    def changed(path, expected):
        calls.append(path)
        if len(calls) > 1:
            raise ValueError("historical bytes changed")
        return original(path, expected)

    monkeypatch.setattr(audit, "read_source", changed)
    with pytest.raises(ValueError, match="historical bytes changed"):
        audit.audit(ref["path"], ref["sha256"], ref["pointer"])
