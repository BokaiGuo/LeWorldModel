import copy
import pytest

from experiments.spectral_reachability import codex_p3_quota_no_shadow_saved_20261030_v1 as analysis
from test_codex_p3_quota_saved_consumer_20261030_v1 import make_events


def test_subset_actual_accounting_and_nonrefund(monkeypatch):
    events, tasks, banks = make_events()
    monkeypatch.setattr(analysis.saved.native, "check_path", lambda task, raw, path: None)
    monkeypatch.setattr(analysis.saved.native, "cost", lambda task, path: -path[1][0][0])
    out = analysis.summarize(events, tasks, banks, [2., 1.])
    assert [a["eligible_candidates_by_root"] for a in out["arms"]] == [[10, 10], [15, 5], [5, 15]]
    assert [a["actual_shadow_candidate_count"] for a in out["arms"]] == [10, 10, 10]
    assert all(a["actual_neural_axis_calls_by_root"] == [30, 30] for a in out["arms"])
    assert out["actual_shadow_not_refunded"] is True
    assert out["subset_counterfactual_is_not_actual_separate_run"] is True
    bad = copy.deepcopy(events)
    next(x for x in bad if x["kind"] == "candidate_complete")["row"]["cost"] = 99
    with pytest.raises(ValueError):
        analysis.summarize(bad, tasks, banks, [2., 1.])
