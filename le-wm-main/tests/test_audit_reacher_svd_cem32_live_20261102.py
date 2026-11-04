import copy

import pytest

from experiments.spectral_reachability import audit_reacher_svd_cem32_live_20261102 as audit

ROOT = audit.experiment.ROOT / "results"
PLAN = ROOT / "reacher_svd_cem32_live_PLAN_20261102_v1.json"
JOURNAL = ROOT / "reacher_svd_cem32_live_RUN_20261102_v1/events.jsonl"
PLAN_SHA = "9b10d01fea866e764978ceac0632e3eda8989a0ed7ba1b81234e5195f46ed088"
JOURNAL_SHA = "38a832235e1f7bfd2163cefe69e5279a13b03f769ac78d1a8ebc2c27d0e70e36"


@pytest.fixture(scope="module")
def events():
    return audit.saved.verify_journal(JOURNAL)


def consume():
    return audit.consume(PLAN, PLAN_SHA, JOURNAL, JOURNAL_SHA)


def test_all_six_arms():
    result = consume()
    assert len(result["results"]) == 6 and result["P1_admission"] is False
    assert [r["role"] for r in result["results"]] == ["selected", "neutral", "same_bank_greedy"] * 2


@pytest.mark.parametrize("mutation", ["stop", "history", "suffix", "missing_arm"])
def test_rejects_tampered_event(monkeypatch, events, mutation):
    altered = copy.deepcopy(events)
    if mutation == "stop":
        next(e for e in altered if e["kind"] == "step_returned")["truncated"] = True
    elif mutation == "history":
        next(e for e in altered if e["kind"] == "context" and e["role"] == "successor")["value"]["history"][2][0] += 1
    elif mutation == "suffix":
        next(e for e in altered if e["kind"] == "arm_returned")["result"]["action_witness_raw"][7][0] += 0.1
    else:
        altered.pop(next(i for i, e in enumerate(altered) if e["kind"] == "arm_started"))
    monkeypatch.setattr(audit.saved, "verify_journal", lambda path: altered)
    with pytest.raises(ValueError):
        consume()
