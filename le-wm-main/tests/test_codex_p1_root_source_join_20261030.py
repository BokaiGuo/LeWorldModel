import copy

import pytest

from experiments.spectral_reachability import codex_p1_root_source_join_20261030 as m


def fixtures():
    old = {"freshness_certified": False, "source_run_preconditions_met": False,
           "used_or_reserved_root_seeds": [10], "used_or_reserved_reset_seeds": [20],
           "reserved_reset_spans": [[100, 200]]}
    extended = {"freshness_certified": False, "P1_admission": False,
                "stable_identity_projections": {"seed_projections": [
                    {"kind": "reset_seed", "seed": 203027101}]}}
    return old, extended


def test_cross_field_seed_collision_and_non_admission():
    old, ext = fixtures()
    result = m.compare(old, ext)
    assert result["candidates"][0]["conflicts"][0]["kind"] == "known_seed"
    assert result["candidates"][0]["conflicts"][0]["candidate_field"] == "root_seed"
    assert result["source_run_authorized"] is False
    assert result["training_disjoint_certified"] is False


def test_half_open_span_and_batch_fixed():
    old, ext = fixtures()
    ext["stable_identity_projections"]["seed_projections"] = []
    old["reserved_reset_spans"] = [[203027201, 203027202]]
    rows = m.compare(old, ext)["candidates"]
    assert rows[0]["conflicts"][0]["kind"] == "known_reset_span"
    assert rows[1]["conflicts"] == []
    assert len(rows) == 4


def test_outcome_irrelevant():
    old, ext = fixtures()
    before = m.compare(old, ext)
    other = copy.deepcopy(ext)
    other["outcome"] = {"success": False, "score": -999}
    assert m.compare(old, other) == before


def test_bad_scope_and_span_rejected():
    old, ext = fixtures()
    old["freshness_certified"] = True
    with pytest.raises(ValueError, match="non-admission"):
        m.compare(old, ext)
    old, ext = fixtures()
    old["reserved_reset_spans"] = [[5, 5]]
    with pytest.raises(ValueError, match="half-open"):
        m.compare(old, ext)
