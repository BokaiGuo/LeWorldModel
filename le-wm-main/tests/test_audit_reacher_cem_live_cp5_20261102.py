import copy

import pytest

from experiments.spectral_reachability import audit_reacher_cem_live_cp5_20261102 as audit

ROOT = audit.experiment.ROOT / "results"
PLAN = ROOT / "reacher_cem_live_cp5_PLAN_20261102_v1.json"
JOURNAL = ROOT / "reacher_cem_live_cp5_RUN_20261102_v1/events.jsonl"
PLAN_SHA = "c9794157724c772b836daf68450efe03d6d4dec5f99638f9f26ba4058ca33764"
JOURNAL_SHA = "3230ae412f44cbbf273e03da8f84b9bf2e18c93463a7636cd3e9daf2d241ba38"


@pytest.fixture(scope="module")
def events():
    return audit.saved.verify_journal(JOURNAL)


def consume():
    return audit.consume(PLAN, PLAN_SHA, JOURNAL, JOURNAL_SHA)


def test_full_saved_output():
    output = consume()
    assert len(output["results"]) == 4 and output["P1_admission"] is False
    assert all(item["cem_candidates"] == 96 and item["prior_final_abs_error"] < 1e-6
               for item in output["results"])


@pytest.mark.parametrize("kind", ["stop", "history", "suffix"])
def test_rejects_corrupted_event(monkeypatch, events, kind):
    altered = copy.deepcopy(events)
    if kind == "stop":
        next(e for e in altered if e["kind"] == "step_returned")["terminated"] = True
    elif kind == "history":
        event = next(e for e in altered if e["kind"] == "context" and e["role"] == "successor")
        event["value"]["history"][2][0] += 1.0
    else:
        event = next(e for e in altered if e["kind"] == "arm_returned")
        event["result"]["action_witness_raw"][6][0] += 0.1
    monkeypatch.setattr(audit.saved, "verify_journal", lambda path: altered)
    with pytest.raises(ValueError):
        consume()
