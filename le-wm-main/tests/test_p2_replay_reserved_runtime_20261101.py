import pytest

from experiments.spectral_reachability import p2_replay_reserved_runtime_20261101 as m


def caps(n=5):
    return {axis: n if axis in m.REPLAY else 1000 for axis in m.base.AXES}


def entry(arm, axis="predict", phase="search"):
    return {"kind": "entry_attempt", "entry": {
        "kind": axis, "samples": 1, "scope": {"arm": arm, "phase": phase}}}


def test_holds_each_arm_and_charges_shared_work():
    events = []
    g = m.ReplayReservedGate(caps(), events.append)
    g.event(entry(m.base.SHARED_ARM))
    assert all(g.attempted[a]["predict"] == 1 for a in m.base.ARMS)
    with pytest.raises(ValueError, match="replay hold"):
        g.event(entry(m.base.ARMS[0]))
    assert g.failed and events[0]["kind"] == "replay_hold"


def test_replay_consumes_hold_only_after_durable_charge():
    events = []
    g = m.ReplayReservedGate(caps(), events.append)
    arm = m.base.ARMS[0]
    for axis in m.REPLAY:
        for _ in range(4):
            g.event(entry(arm, axis, "replay"))
    assert g.remaining[arm] == {"predict": 0, "action_encoder": 0}
    assert g.attempted[arm]["predict"] == 4
    assert g.remaining[m.base.ARMS[1]] == m.REPLAY
    with pytest.raises(ValueError, match="unexpected replay"):
        g.event(entry(arm, phase="replay"))


def test_failed_journal_poisoned_before_dispatch_and_no_refund():
    events = []

    def sink(event):
        events.append(event)
        if event["kind"] == "entry_attempt":
            raise OSError("journal failed")

    g = m.ReplayReservedGate(caps(), sink)
    with pytest.raises(OSError):
        g.event(entry(m.base.ARMS[0], phase="replay"))
    assert g.failed
    assert g.attempted[m.base.ARMS[0]]["predict"] == 1
    assert g.remaining[m.base.ARMS[0]]["predict"] == 4
    with pytest.raises(ValueError, match="poisoned"):
        g.event(entry(m.base.ARMS[0]))


def test_invalid_reserve_and_cpu_entry_fail_before_output(tmp_path):
    with pytest.raises(ValueError, match="reserve"):
        m.ReplayReservedGate(caps(3), lambda _: None)
    with pytest.raises(ValueError, match="host CUDA"):
        m.run_native(model=None, device=None, ready=None, compiled=None,
                     provenance_by_arm=None, output=tmp_path / "run", caps=caps())
    assert not (tmp_path / "run").exists()
