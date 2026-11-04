import copy

import pytest

from experiments.spectral_reachability import p3_all30_cpu_consume_20261102_codexexclusive as consume

ROOT = consume.replay.ROOT
PREP = ROOT / "results/p3_all30_cpu_physical_20261102_codexexclusive_PREP_01.json"
RUN = ROOT / "results/p3_all30_cpu_physical_20261102_codexexclusive_RUN_01"
PREP_SHA = "375da1e30a17cde98fb60879d0649c42a8e5f283ec2d0c56c702efdc27e4aadb"
REPORT_SHA = "8dfa0ece3f13972cf305eacb6f53e265b1b0713feca3b6038ca5473526dba704"


@pytest.fixture(scope="module")
def record():
    p, data = consume.replay.verify(PREP, PREP_SHA)
    report = consume.moving.read(RUN / "report.json")
    events = consume.chain(RUN / "journal.jsonl", report["journal_chain_tail"])
    return p, data, report, events


def test_actual_partial_result():
    value = consume.consume(PREP, PREP_SHA, RUN, REPORT_SHA)
    assert (value["completed_branches"], value["failed_branches"], value["not_started"]) == (7, 1, 22)
    assert value["png_verified"] == 174


@pytest.mark.parametrize("fault", ["action", "guard", "post_stop", "resource", "source_state"])
def test_reject_corrupted_partial_records(record, fault):
    p, data, report, events = copy.deepcopy(record)
    if fault == "action":
        next(e for e in events if e["kind"] == "returned_transition" and e["phase"] == "plan")["command"][0] = 5
    elif fault == "guard":
        next(e for e in events if e["kind"] == "guard_receipt" and e["branch"] == 0)["accepted"] = False
    elif fault == "post_stop":
        events.append({"kind": "unexpected", "branch": 8})
    elif fault == "resource":
        report["counts"]["public_step"]["completed"] += 1
    else:
        data["references"][0]["states"][0][0] += 1
    with pytest.raises((ValueError, KeyError)):
        consume.validate(p, data, report, events, RUN)
