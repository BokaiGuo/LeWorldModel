import hashlib
import json
import math

import pytest

from experiments.spectral_reachability import p1_known_history_exclusion_20261101_hypatia as m


@pytest.fixture(scope="module")
def actual():
    return m.generate()


def test_current_pinned_evidence_and_unknowns(actual):
    assert len(actual["historical_identities"]) == 84
    assert len(actual["seed_projections"]) == 40
    assert len(actual["reserved_half_open_reset_spans"]) == 18
    assert actual["history_complete"] is False
    assert actual["development_disjoint_certified"] is False
    assert actual["training_disjoint_certified"] is False
    assert actual["freshness_certified"] is False
    assert actual["P1_admission"] is False
    assert actual["new_candidates_selected"] == 0


def test_pose_wrap_and_signed_zero_are_not_evidence_of_new_root():
    a = m.identity([0, -0.0, 2, 3, 0, 1, 2])
    b = m.identity([-0.0, 0, 2, 3, 2 * math.pi, 4, 5])
    assert a["pose_sha256"] == b["pose_sha256"]
    assert a["state_sha256"] != b["state_sha256"]


def test_exclusive_create_verify_and_tamper(tmp_path, monkeypatch, actual):
    monkeypatch.setattr(m, "generate", lambda: actual)
    path = tmp_path / (m.NAME + "_test.json")
    digest = m.create(path)
    assert m.verify(path, digest)["freshness_certified"] is False
    with pytest.raises(FileExistsError):
        m.create(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        m.verify(path, digest)
    with pytest.raises(ValueError, match="drift"):
        m.verify(path, hashlib.sha256(b"{}").hexdigest())


def test_bad_or_changed_source_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"a":1}')
    with pytest.raises(ValueError, match="SHA"):
        m.read_bound(path, "0" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        m.decode(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="constant"):
        m.decode(b'{"a":NaN}')


def test_no_outcome_fields_in_projected_identities(actual):
    for row in actual["historical_identities"]:
        assert not ({"outcome", "success", "cost", "reward"} & set(row))
    assert all("evidence" in row for row in actual["root_goal_groups"])
    assert json.loads(m.encode(actual)) == actual
