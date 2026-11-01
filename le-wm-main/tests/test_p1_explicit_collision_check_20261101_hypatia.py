import json
import math

import pytest

from experiments.spectral_reachability import p1_explicit_collision_check_20261101_hypatia as m


@pytest.fixture(scope="module")
def history():
    return m.base.decode(m.base.read_bound(m.HISTORY_PATH, m.HISTORY_SHA))


@pytest.fixture(scope="module")
def supplement():
    return m.base.decode(m.base.read_bound(m.COHERENT_PATH, m.COHERENT_SHA))


def document(*rows):
    return {"schema": m.NAME + ".candidates.v1", "candidates": list(rows)}


def candidate(label, root=None, reset=None, state=None):
    return {"candidate_id": label, "root_seed": root, "reset_seed": reset, "state7": state}


def test_known_state_pose_and_seed_collision(history, supplement):
    state = supplement["record"]["state7"]
    report = m.compare(document(candidate("replay", root=supplement["record"]["reset_seed"], state=state)),
                       history, supplement)[0]
    assert report["collision_observed"] is True
    assert {"known_seed", "exact_state7_bits", "pose_numeric_alias"} <= {
        hit["kind"] for hit in report["known_history_hits"]}
    assert any(hit["reference"] == "coherent:/record/state7" for hit in report["known_history_hits"])
    assert report["freshness"] == "unknown"


def test_span_is_half_open_and_compares_both_seed_fields(history, supplement):
    lo, hi = history["reserved_half_open_reset_spans"][0]
    custom = dict(history, reserved_half_open_reset_spans=[[lo, hi]],
                  seed_projections=[], inventory8_root_seeds=[], inventory8_reset_seeds=[])
    result = m.compare(document(candidate("inside", root=lo), candidate("end", reset=hi)),
                       custom, supplement)
    assert any(x["kind"] == "reserved_reset_span" for x in result[0]["known_history_hits"])
    assert not any(x["kind"] == "reserved_reset_span" for x in result[1]["known_history_hits"])


def test_batch_alias_wrap_velocity_and_cross_seed(history, supplement):
    s = [17., 18., 19., 20., 0., 0., 0.]
    other = [17., 18., 19., 20., 2 * math.pi, 8., 9.]
    rows = m.compare(document(candidate("a", root=1234, state=s),
                              candidate("b", reset=1234, state=other)), history, supplement)
    for row in rows:
        assert {x["kind"] for x in row["batch_hits"]} == {"candidate_seed_collision", "candidate_pose_numeric_alias"}


def test_missing_is_unknown_never_fresh(history, supplement):
    row = m.compare(document(candidate("unobserved", root=9999999999)), history, supplement)[0]
    assert not row["collision_observed"]
    assert row["missing_identity_fields"] == ["reset_seed", "state7"]
    assert row["development_history_independence"] == row["freshness"] == row["training_disjoint"] == "unknown"


@pytest.mark.parametrize("entry", [
    {"candidate_id": "x", "root_seed": 1, "reset_seed": None, "state7": None, "success": True},
    candidate("x", root=True),
    candidate("x", state=[0] * 6),
    candidate("x", state=[0, 0, 0, 0, float("nan"), 0, 0]),
    candidate("x"),
])
def test_reject_extra_outcomes_or_invalid_identity(entry):
    with pytest.raises(ValueError):
        m.validate_candidates(document(entry))


def test_create_verify_candidate_tamper_and_no_overwrite(tmp_path):
    path = tmp_path / "explicit.json"
    path.write_text(json.dumps(document(candidate("fixture_only", root=9999999999))))
    digest = m.base.sha(path.read_bytes())
    out = tmp_path / (m.NAME + "_test.json")
    result_sha = m.create(path, digest, out)
    assert m.verify(out, result_sha)["freshness_certified"] is False
    with pytest.raises(FileExistsError):
        m.create(path, digest, out)
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        m.verify(out, result_sha)


def test_candidate_ids_unique():
    with pytest.raises(ValueError, match="unique"):
        m.validate_candidates(document(candidate("same", root=1), candidate("same", root=2)))
