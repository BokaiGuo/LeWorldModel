import copy

import pytest

from experiments.spectral_reachability import p1_uphill_observed_pairing_table_20261031 as audit


@pytest.fixture(scope="module")
def inputs():
    return [audit.pairing.read_checked(path, sha) for path, sha in (
        (audit.common.PATH, audit.common.SHA),
        (audit.COMMON_PATH, audit.COMMON_SHA),
        (audit.PLAN_PATH, audit.PLAN_SHA))]


def test_join_preserves_censoring_and_recovers_observed_success(inputs):
    rows = audit.join(*inputs)
    assert len(rows) == 1008
    assert sum(r["observed_common_h10"] for r in rows) == 1006
    recovered = [r for r in rows if r["common_status"] == "native_stopped"
                 and r["observed_common_h10"]]
    assert len(recovered) == 2
    assert all(r["observed_common_h10_native_success"] for r in recovered)
    assert all(r["common_endpoint_object_position_error"] is None for r in recovered)
    censored = [r for r in rows if not r["observed_common_h10"]]
    assert len(censored) == 2
    assert all(r["observed_common_h10_object_position_error"] is None for r in censored)


def test_rejects_tampering_and_mismatched_plan_identity(inputs):
    data, common_data, plan_data = copy.deepcopy(inputs)
    common_data["rows"][0]["role"] = "bad"
    with pytest.raises(ValueError):
        audit.join(data, common_data, plan_data)
    data, common_data, plan_data = copy.deepcopy(inputs)
    plan_data["rows"][0]["logical_pair"] = 999
    with pytest.raises(ValueError):
        audit.join(data, common_data, plan_data)
