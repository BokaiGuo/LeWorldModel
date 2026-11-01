from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_p3_six_physical_resource_review_20261030_v1 as review


@pytest.fixture(scope="module")
def records():
    report = review.saved.host.helper.read(review.RUN / "report.json")
    events = review.saved.kernel.journal(review.RUN / "journal.jsonl", report["journal_chain_tail"])
    binding = review.saved.host.helper.read(review.BIND)
    return report, events, binding


def test_actual_six_slot_counts(records):
    rows = review.reconcile(*records)
    assert len(rows) == 6
    assert all(r["physical"]["physics_step"] == {"attempted": 204, "completed": 204} for r in rows)
    assert [r["prior_metric_returns"] for r in rows] == [10, 15, 5, 10, 5, 15]


@pytest.mark.parametrize("kind", ["branch", "identity", "missing", "action", "total"])
def test_resource_tampering_rejected(records, kind):
    report, events, binding = deepcopy(records)
    event = next(e for e in events if e["kind"] == "operation_completed")
    if kind == "branch":
        event["branch"] = 6
    elif kind == "identity":
        event["arm"] = "wrong"
    elif kind == "missing":
        events.remove(event)
    elif kind == "action":
        binding["rows"][0]["raw_float32_sha256"] = "wrong"
    else:
        report["counts"]["public_step"]["completed"] -= 1
    with pytest.raises(ValueError):
        review.reconcile(report, events, binding)
