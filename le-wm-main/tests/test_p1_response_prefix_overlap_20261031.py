import copy

import pytest

from experiments.spectral_reachability import p1_response_prefix_overlap_20261031 as audit


@pytest.fixture
def sources():
    return (audit.pairing.read_checked(audit.PANEL, audit.PANEL_SHA),
            audit.pairing.read_checked(audit.PAIRS, audit.PAIRS_SHA))


def test_all_candidates_are_accounted_for_without_outcomes(sources):
    panel, pairs = sources
    result = audit.audit(panel, pairs)
    assert len(result["rows"]) == 54
    assert result["matched_count"] == 14
    assert result["physical_execution"] is False
    assert all("success" not in row for row in result["rows"])


def test_rejects_invalid_prefix(sources):
    panel, pairs = sources
    panel = copy.deepcopy(panel)
    first = next(iter(panel["rows"][0]["candidates"].values()))
    first["prefix_float32_hex"] = "00"
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        audit.audit(panel, pairs)


def test_rejects_missing_matched_anchor(sources):
    panel, pairs = sources
    pairs = copy.deepcopy(pairs)
    pairs["anchors"][0]["identity"]["arm"] = "invalid"
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        audit.audit(panel, pairs)
