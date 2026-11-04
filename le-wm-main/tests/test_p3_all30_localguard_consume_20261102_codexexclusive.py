import copy

import pytest

from experiments.spectral_reachability import p3_all30_localguard_consume_20261102_codexexclusive as consumer

ROOT = consumer.run.ROOT
PREP = ROOT / "results/p3_all30_localguard_cpu_20261102_codexexclusive_PREP_01.json"
RUN = ROOT / "results/p3_all30_localguard_cpu_20261102_codexexclusive_RUN_01"
PREP_SHA = "5383802644f3abb8409bebe3d3fa53263f6d06e4c5d347e49e5ef7d757667c7d"
REPORT_SHA = "059b85e429785712da326169a5ed80c5d67471434ae776e5ecdacafb71b09c99"


@pytest.fixture(scope="module")
def record():
    p, data = consumer.run.verify(PREP, PREP_SHA)
    report = consumer.moving.read(RUN / "report.json")
    events = consumer.prior.chain(RUN / "journal.jsonl", report["journal_chain_tail"])
    return p, data["references"], report, events


def test_real_all30():
    result = consumer.consume(PREP, PREP_SHA, RUN, REPORT_SHA)
    assert result["status"] == "all_records_verified"
    assert result["fixed_denominator"] == 30
    assert result["valid_endpoint_count"] == 26
    assert result["candidate_counts"]["guard_failed_invalid"] == 4
    assert result["png_verified"] == 655


@pytest.mark.parametrize("fault", ["command", "invalid_guard", "unstarted", "denominator", "resource"])
def test_reject_tampering(record, fault):
    p, refs, report, events = copy.deepcopy(record)
    if fault == "command":
        next(e for e in events if e["kind"] == "returned_transition" and e["phase"] == "plan")["command"][0] = 9
    elif fault == "invalid_guard":
        next(e for e in events if e["kind"] == "guard_receipt" and not e["accepted"])["accepted"] = True
    elif fault == "unstarted":
        report["rows"][8]["status"] = "not_started"
    elif fault == "denominator":
        report["valid_endpoint_count"] = 30
    else:
        report["counts"]["public_step"]["completed"] += 1
    with pytest.raises(ValueError):
        consumer.validate(p, refs, report, events, RUN)
