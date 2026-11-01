import copy

import pytest

from experiments.spectral_reachability import p1_exploration_run01_receipt_20261101_hypatia as mod


def actual():
    request = mod.prior.base.decode(mod.prior.base.read_bound(mod.REQUEST, mod.INPUTS[mod.REQUEST]))
    report = mod.prior.base.decode(mod.prior.base.read_bound(mod.DIRECTORY / "sourcepack.json",
                                                           mod.INPUTS[mod.DIRECTORY / "sourcepack.json"]))
    return request, report, mod.runner.read_journal(mod.DIRECTORY / "journal.jsonl")


def test_real_receipt_contains_all_reset_states():
    result = mod.build()
    assert len(result["reset_comparisons"]) == 16
    assert len(result["selected_sources"]) == 2
    assert result["freshness"] == "unknown"
    assert result["P1_admission"] is False


def test_request_tamper_is_rejected():
    request, report, events = actual()
    request["plan"]["placement_reset_seed"] = 123
    with pytest.raises(ValueError, match="request binding"):
        mod.check_binding(request, report, events)


def test_incomplete_journal_is_rejected():
    request, report, events = actual()
    altered = copy.deepcopy(events)
    altered.pop(next(i for i, e in enumerate(altered) if e["kind"] == "operation_completed"))
    with pytest.raises(ValueError):
        mod.check_binding(request, report, altered)


def test_attempt_receipt_tamper_is_rejected():
    request, report, events = actual()
    report["slots"][0]["proposals"][0]["attempts"][0]["state"][0] += 1
    with pytest.raises(ValueError):
        mod.check_binding(request, report, events)
