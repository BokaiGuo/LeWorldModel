import pytest

from experiments.spectral_reachability.codex_p1_pusher_latent_pairs_20261030_main import build


def test_reject_incomplete_roster():
    with pytest.raises(ValueError, match="fixed panel size"):
        build({"rows": []}, {"rows": []})


def test_reject_mismatched_identity():
    e = {"index": 0, "status": "probe_completed", "returned_suffix": 10,
         "identity": {"root_slot": 0, "condition": "off", "role": "selected", "logical_pair": 0}}
    p = {"index": 0, "status": "probe_completed", "returned_suffix": 10,
         "root_slot": 1, "condition": "off", "role": "selected", "logical_pair": 0}
    with pytest.raises(ValueError, match="identity mismatch"):
        build({"rows": [e] * 24}, {"rows": [p] * 24})


def test_early_stops_are_explicitly_retained_without_imputation():
    rows = [{"index": i, "status": "native_stopped", "returned_suffix": 4} for i in range(24)]
    result = build({"rows": rows}, {"rows": rows})
    assert result["completed_rows"] == 0
    assert len(result["excluded"]) == 24
    assert result["pairs"] == []
