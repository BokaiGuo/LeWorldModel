from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p2_fixed_call_budget_20261030_v1 as m


def events():
    output = []
    for arm, count in [(m.SHARED, 4), (m.ARMS[0], 164), (m.ARMS[1], 164)]:
        for kind in ("predict", "action_encoder"):
            output.extend({"kind": "entry_terminal", "entry": {
                "scope": {"arm": arm}, "kind": kind, "samples": 1, "completed": True}}
                for _ in range(count))
    return output


def test_shared_charged_twice_but_unique_once():
    checked = m.inspect_events(events())
    assert checked["unique"] == {"predict": 332, "action_encoder": 332}
    assert all(checked["attributed"][arm] == {"predict": 168, "action_encoder": 168}
               for arm in m.ARMS)


@pytest.mark.parametrize("damage", ["missing", "failed", "wrong_owner", "wrong_sample"])
def test_bad_entry_fails_closed(damage):
    records = events()
    if damage == "missing":
        records.pop()
    elif damage == "failed":
        records[0]["entry"]["completed"] = False
    elif damage == "wrong_owner":
        records[0]["entry"]["scope"]["arm"] = "other"
    else:
        records[0]["entry"]["samples"] = 2
    with pytest.raises(ValueError):
        m.inspect_events(records)


def test_existing_four_roots_real_cost():
    result = m.audit_run(m.ROOT / "results/codex_p2_knownroot_host_cuda_20261030_v1_run01")
    assert len(result["rows"]) == 4 and result["unique_calls_per_kind"] == 1328
    assert all(r["selection_evaluations"] == [80, 40] for r in result["rows"])
    assert result["equal_total_compute"] is False


def test_execution_requires_ack_before_manifest_or_backend(tmp_path):
    with pytest.raises(ValueError, match="acknowledgement"):
        m.run("/missing", "x", tmp_path / "new", "wrong")
    assert not (tmp_path / "new").exists()
