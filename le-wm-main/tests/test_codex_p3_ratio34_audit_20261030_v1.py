from copy import deepcopy
import pytest
from experiments.spectral_reachability import codex_p3_ratio34_audit_20261030_v1 as a


def fixture():
    actual = {axis: 0 for axis in a.AXES}
    actual["metric"] = 2
    row = dict(point=20, arm="uniform", allocation=[1, 1], actual=actual,
               caps={axis: 10 for axis in a.AXES}, attempt={"wall_seconds": 1.})
    events = []
    winners = []
    for root in (0, 1):
        entry = dict(index=root, resource="metric", scope=[20, "uniform", "search", root],
                     completed=False, started_seconds=1.)
        events.append(dict(kind="resource_attempt", entry=entry))
        events.append(dict(kind="resource_return", entry=dict(entry, completed=True, ended_seconds=2.)))
        winner = dict(index=0, cost=float(root), commands=[[0., 0.]])
        winners.append(winner)
        events.append(dict(kind="candidate_complete", scope=entry["scope"], candidate=winner))
    receipt = dict(status="semantic_saved_checked_not_scientifically_admitted",
                   rows=[dict(point=20, arm="uniform", attempted=actual, allocation=[1, 1], winners=winners)])
    return events, receipt, row


def test_reconcile_rejects_missing_reorder_and_forged_winner():
    events, receipt, row = fixture()
    assert a.reconcile(events, receipt, row)["winner_indices"] == [0, 0]
    broken = deepcopy(events)
    broken.pop(1)
    with pytest.raises(ValueError):
        a.reconcile(broken, receipt, row)
    broken = deepcopy(events)
    broken[2]["candidate"]["index"] = 1
    with pytest.raises(ValueError):
        a.reconcile(broken, receipt, row)
    broken = deepcopy(receipt)
    broken["rows"][0]["winners"][0]["cost"] = 99.
    with pytest.raises(ValueError):
        a.reconcile(events, broken, row)
