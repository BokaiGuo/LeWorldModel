from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_total_cost_reconcile_20261101_leibniz as audit


def fixture():
    op = dict(operation_id=0, episode_id="s0", resource="encoder", units=1, region="control")
    counts = {"encoder": {"attempted": 1, "completed": 1}}
    return ([dict(kind="operation_attempted", **op), dict(kind="operation_completed", **op),
             dict(kind="run_finished", counts=counts)],
            dict(counts=counts, pending={}))


def test_reconciles_actual_rgb_without_node_zero_substitution():
    events, report = fixture()
    assert audit.physical_census(events, report)["encoder"] == {"attempted": 1, "completed": 1}


@pytest.mark.parametrize("mutation", ["missing_return", "duplicate_attempt", "wrong_identity", "tamper_counts"])
def test_rejects_bad_operations(mutation):
    events, report = deepcopy(fixture())
    if mutation == "missing_return":
        events.pop(1)
    elif mutation == "duplicate_attempt":
        events.insert(1, deepcopy(events[0]))
    elif mutation == "wrong_identity":
        events[1]["episode_id"] = "other"
    else:
        report["counts"]["encoder"]["completed"] = 2
    with pytest.raises(ValueError):
        audit.physical_census(events, report)


def test_pending_is_unknown_completion_not_zero_attempt():
    events, report = fixture()
    events.pop(1)
    op = {k: events[0][k] for k in ("operation_id", "episode_id", "resource", "units", "region")}
    report["pending"] = {"0": op}
    report["counts"] = {"encoder": {"attempted": 1, "completed": 0}}
    events[-1]["counts"] = deepcopy(report["counts"])
    assert audit.physical_census(events, report)["encoder"] == {"attempted": 1, "completed": 0}


def test_fixed_roster_and_hash_binding():
    assert set(audit.ROSTER) == {"root0_structured", "root0_cem", "root1_cem"}
    name, (folder, _) = next(iter(audit.ROSTER.items()))
    with pytest.raises(ValueError, match="pinned report drift"):
        audit.reconcile(name, (folder, "0" * 64))
