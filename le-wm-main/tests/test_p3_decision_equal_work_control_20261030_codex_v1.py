from copy import deepcopy

import pytest

import agent_p3_fullcommit_live_20261028_v1_tests as fixture
from test_p3_feedback_reverse_control_20261030_v1 import Session, setup
from experiments.spectral_reachability import p3_decision_equal_work_control_20261030_codex_v1 as m
from experiments.spectral_reachability import agent_p3_shared_panel_summary_20261029_v1 as shared_summary


def run(policy, session=None, backend=fixture.RamBackend):
    root, prior = setup()
    events = fixture.Events()
    result = m.execute_slot(root, policy, prior, session or Session(),
                            lambda task, store: backend(task, store), events, permit_ram=True)
    return result, events.events


def test_four_real_callback_decisions_pay_both_panels_before_each_packet():
    signal, events = run("signal")
    reverse, _ = run("reverse_signal")
    fixed, _ = run("fixed_h1")
    for slot in (signal, reverse, fixed):
        assert slot["status"] == "callback_complete_unverified", slot["error"]
        assert [p["end"] for p in slot["packets"]] == [5, 10, 15, 20]
        assert all(d["resource_receipt"]["attempted"] == d["resource_receipt"]["completed"] == m.expected()
                   for d in slot["decisions"])
    assert [e["kind"] for e in events if e["kind"] in ("decision_paid", "packet_attempt")] == [
        "decision_paid", "packet_attempt"] * 4
    compared = m.compare(signal, reverse)
    assert [x["step"] for x in compared["per_decision"]] == [0, 5, 10, 15]
    assert all(x["measured_vector_match"] and x["hook_shape_multiset_match"] for x in compared["per_decision"])
    assert not compared["scientific_pass"] and not compared["physical_certified"]


@pytest.mark.parametrize("h", [1, 3])
def test_routing_and_common_fail_stop(h):
    decision = dict(continue_expanding=True, program_type="progress", extension_type="progress", horizon=h)
    assert m.choose("signal", decision) == h
    assert m.choose("reverse_signal", decision) == 4 - h
    assert m.choose("fixed_h1", decision) == 1
    for policy in m.POLICIES:
        with pytest.raises(ValueError):
            m.choose(policy, dict(decision, continue_expanding=False))


def test_failure_cost_is_not_refunded_or_executed():
    class Broken(fixture.RamBackend):
        def replay(self, *args):
            raise RuntimeError("failed")
    result, _ = run("signal", backend=Broken)
    assert result["status"] == "failed" and not result["packets"]
    receipt = result["failed_decision_receipt"]
    assert receipt["attempted"]["replay_calls"] == 1 and receipt["completed"]["replay_calls"] == 0
    m.core.audit_ledger(receipt)


def test_stop_and_encoder_failure_are_not_padded():
    class Stop(Session):
        def packet(self, start, raw):
            return dict(raw=raw[:2].copy(), rgb_bytes=b"", end=start + 2, stopped=True, episode_id=self.episode_id)
    result, _ = run("reverse_signal", Stop())
    assert result["status"] == "native_stopped" and len(result["decisions"]) == 1
    assert len(result["packets"]) == 1 and result["packets"][0]["end"] == 2
    class Broken(Session):
        def encode(self, end, rgb):
            raise RuntimeError("encode")
    failed, _ = run("signal", Broken())
    assert failed["status"] == "failed" and len(failed["decisions"]) == 1 and len(failed["packets"]) == 1
    assert failed["callback_attempted"] == {"packet": 1, "encode": 1}
    assert failed["callback_completed"] == {"packet": 1, "encode": 0}
    assert m.compare(failed, result)["per_decision"] is None


@pytest.mark.parametrize("kind", ["cap", "decision", "stage", "root", "packet", "actions", "callbacks"])
def test_tampered_receipt_or_binding_rejected(kind):
    a, _ = run("signal")
    b, _ = run("reverse_signal")
    b = deepcopy(b)
    if kind == "cap":
        b["decisions"][1]["resource_receipt"]["caps"]["predict_API"] += 1
    elif kind == "decision":
        b["decisions"][1]["selected_horizon"] = 4 - b["decisions"][1]["selected_horizon"]
    elif kind == "stage":
        b["decisions"][1]["resource_receipt"]["events"][0]["scope"]["stage"] = 0
    elif kind == "root":
        b["root_sha256"] = "bad"
    elif kind == "packet":
        b["packets"][1]["end"] = 11
    elif kind == "actions":
        b["packets"][1]["raw"][0][0] += 0.01
    else:
        b["callback_completed"]["encode"] = 3
    with pytest.raises(ValueError):
        m.compare(a, b)


def test_fixed_roster_keeps_setup_failure_and_cleanup():
    root, prior = setup()
    closed = []
    def factory(slot, root_index, policy):
        if slot == 2:
            raise RuntimeError("setup")
        return Session(), prior, lambda: closed.append(slot)
    result = m.execute_roster([root, root], factory, lambda task, store: fixture.RamBackend(task, store),
                              fixture.Events(), permit_ram=True)
    assert result["fixed_denominator"] == 6
    assert [(r["slot"], r["root"], r["policy"]) for r in result["rows"]] == m.ROSTER
    assert result["rows"][2]["status"] == "setup_failed" and closed == [0, 1, 3, 4, 5]
    assert all(r["status"] == "callback_complete_unverified" for r in result["rows"] if r["slot"] != 2)


def test_frozen_caps_match_existing_shared_panel_definition():
    assert m.expected() == shared_summary.node_costs()
