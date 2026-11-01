import copy

import pytest

from experiments.spectral_reachability import p1_call_evidence_join_20261101_hypatia as mod


def inputs():
    reader = mod.authority.Reader()
    catalog = mod.authority.parse(reader.read(str(mod.authority.CATALOG), mod.authority.CATALOG_SHA,
                                              mod.authority.CAP))
    recovery = mod.authority.parse(reader.read(str(mod.RECOVERY), mod.RECOVERY_SHA, mod.authority.CAP))
    return mod.authority.requests(catalog), recovery


def test_actual_fixed_inputs_join_ten_calls():
    report = mod.generate()
    assert len(report["calls"]) == 10
    assert sum(r["source_id"] == "B" for r in report["calls"]) == 3
    assert all(r["reset_attempt_ledger"] == "not_established" for r in report["calls"])
    assert report["freshness_certified"] is False


def test_reject_call_and_family_mismatch():
    request, recovery = inputs()
    recovery = copy.deepcopy(recovery)
    recovery["calls"][0]["reported_family_source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        mod.join(request, recovery)


def test_reject_duplicate_call():
    request, recovery = inputs()
    recovery["calls"][0]["call_id"] = recovery["calls"][1]["call_id"]
    with pytest.raises(ValueError):
        mod.join(request, recovery)
