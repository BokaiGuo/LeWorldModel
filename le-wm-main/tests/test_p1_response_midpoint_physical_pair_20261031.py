import copy

import pytest

from experiments.spectral_reachability import p1_response_midpoint_physical_pair_20261031 as pair


def fixture():
    read = pair.experiment.prior.bundle.physical.prior.native.read
    prepared = read(pair.PREP, cap=64 * 1024**2)
    original = read(pair.experiment.source.RECEIPT, cap=64 * 1024**2)
    by_id = {row["index"]: row for row in original["rows"]}
    current = {"status": "saved_checks_passed", "fixed_rows": 16, "excluded": prepared["excluded"],
               "rows": [copy.deepcopy(by_id[item["index"]]) for item in prepared["roster"]]}
    return prepared, original, current


def test_pair_preserves_exact_endpoint_and_censored_status():
    prepared, old, new = fixture()
    result = pair.pair(prepared, old, new)
    assert len(result["rows"]) == 13
    assert all(row["midpoint_replanned_minus_baseline"]["object_position_error"] == 0
               for row in result["rows"])
    new["rows"][0]["status"] = "native_stopped"
    new["rows"][0]["completed_endpoint"] = None
    result = pair.pair(prepared, old, new)
    assert result["rows"][0]["midpoint_replanned_endpoint"] is None
    assert result["rows"][0]["midpoint_replanned_minus_baseline"] is None
    assert result["equal_budget_certified"] is False


def test_pair_rejects_identity_and_imputed_endpoint():
    prepared, old, new = fixture()
    new["rows"][0]["arm"] = "different"
    with pytest.raises(Exception, match="identity"):
        pair.pair(prepared, old, new)
    prepared, old, new = fixture()
    new["rows"][0]["status"] = "censored_out_of_domain"
    with pytest.raises(Exception, match="censor"):
        pair.pair(prepared, old, new)
