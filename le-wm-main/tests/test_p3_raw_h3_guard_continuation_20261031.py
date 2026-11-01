import copy

import pytest

from experiments.spectral_reachability import p3_raw_h3_guard_continuation_20261031 as continuation


def test_real_guard_partial_and_reject_unrelated_failure():
    directory = continuation.PREVIOUS
    report = continuation.host.helper.read(directory / "report.json")
    events = continuation.source.batch.kernel.journal(directory / "journal.jsonl",
                                                      report["journal_chain_tail"])
    assert continuation.allowed_partial(report, events) == 2
    changed = copy.deepcopy(report)
    changed["rows"][2]["first_error"]["message"] = "other failure"
    with pytest.raises(ValueError, match="not certified canvas guard"):
        continuation.allowed_partial(changed, events)
    changed = copy.deepcopy(report)
    changed["rows"][2]["last_observation"]["state"][1] = 500
    changed["rows"][2]["last_observation"]["observation_state"][1] = 500
    with pytest.raises(ValueError, match="not certified canvas guard"):
        continuation.allowed_partial(changed, events)
    changed = copy.deepcopy(events)
    next(e for e in changed if e["kind"] == "guard_receipt"
         and e.get("branch") == 2 and e.get("accepted") is False)["step"] += 1
    with pytest.raises(ValueError, match="guard event binding"):
        continuation.allowed_partial(report, changed)
