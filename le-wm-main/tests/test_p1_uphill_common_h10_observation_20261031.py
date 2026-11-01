import copy

import pytest

from experiments.spectral_reachability import p1_uphill_common_h10_observation_20261031 as audit


def test_recover_only_observed_tenth_steps_and_keep_original_comparisons():
    data = audit.source.read_checked(audit.PATH, audit.SHA)
    rows = audit.build(data)
    assert len(rows) == 1008
    assert sum(r["h10_metrics"] is not None for r in rows) == 1006
    recovered = [r for r in rows if r["status"] == "native_stopped" and r["h10_metrics"]]
    assert [(r["logical_pair"], r["role"], r["probe"]) for r in recovered] == [
        (2, "selected", "original_selected_tail"), (3, "selected", "original_selected_tail")]
    assert all(r["h10_metrics"]["native_success"] for r in recovered)
    original = {(p["logical_pair"], p["probe"], role): value
                for p in data["common_suffix_pairs"] for role, value in p["roles"].items()}
    for row in rows:
        old = original[(row["logical_pair"], row["probe"], row["role"])]
        if row["status"] == "probe_completed":
            assert row["h10_metrics"] == old["endpoint"]
            if old["endpoint_minus_zero_prefix"] is not None:
                assert row["endpoint_minus_zero_prefix"] == old["endpoint_minus_zero_prefix"]


def test_reject_invalid_terminal_status_and_omitted_comparison():
    data = copy.deepcopy(audit.source.read_checked(audit.PATH, audit.SHA))
    target = next(p for p in data["common_suffix_pairs"]
                  if p["logical_pair"] == 2 and p["probe"] == "original_selected_tail")
    target["roles"]["selected"]["last_accepted_terminated"] = False
    with pytest.raises(ValueError):
        audit.build(data)
    data = copy.deepcopy(audit.source.read_checked(audit.PATH, audit.SHA))
    data["common_suffix_pairs"].pop()
    with pytest.raises(ValueError):
        audit.build(data)
