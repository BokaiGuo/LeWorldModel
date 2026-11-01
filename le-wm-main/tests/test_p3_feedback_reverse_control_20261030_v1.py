"""RAM callbacks only, not a native host or scientific evidence."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

import agent_p3_fullcommit_live_20261028_v1_tests as fixture
import test_p3_matched_order_feedback_seam_codex_20261029 as original_fixture
from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as m


class Session:
    episode_id = "fixture"

    def packet(self, start, raw):
        return dict(raw=raw.copy(), rgb_bytes=f"frame{start + 5}".encode(),
                    end=start + 5, stopped=False, episode_id=self.episode_id)

    def encode(self, end, rgb):
        return dict(latent=np.full(192, end / 100, np.float32), input_sha256=hashlib.sha256(rgb).hexdigest(),
                    end=end, episode_id=self.episode_id)


def setup():
    root, _ = fixture.request()
    warm = fixture.m.old.kernel.moving.warmup_commands()
    root["root_context"]["action_history"] = fixture.m.moving.normalized(root, warm)
    root["binding"]["context_sha256"] = m.core.digest(root["root_context"])
    return root, warm[-5:].copy()


def run(policy, session=None, backend=fixture.RamBackend):
    root, prior = setup()
    events = fixture.Events()
    result = m.execute_slot(root, policy, prior, session or Session(),
                            lambda task, store: backend(task, store), events, permit_ram=True)
    return result, events.events


@pytest.mark.parametrize("horizon", [1, 3])
def test_reverse_is_complement_of_admissible_original_only(horizon):
    decision = dict(continue_expanding=True, program_type="progress", extension_type="progress", horizon=horizon)
    assert m.choose("signal_order", decision, [1, 3], 0) == horizon
    assert m.choose("reverse_signal_order", decision, [1, 3], 0) == 4 - horizon
    assert m.choose("fixed_order", decision, [1, 3], 0) == 1
    assert m.choose("reverse_signal_order", decision, [horizon], 1) == horizon
    for update in ({"continue_expanding": False}, {"program_type": "reconfiguration"},
                   {"extension_type": "reconfiguration"}, {"horizon": 2}):
        with pytest.raises(ValueError):
            m.choose("reverse_signal_order", dict(decision, **update), [1, 3], 0)


def test_three_callbacks_same_measured_end_budget_and_live_context():
    values = [run(policy)[0] for policy in m.POLICIES]
    for result in values:
        assert result["status"] == "callback_feedback_complete_not_certified", result["errors"]
        assert result["resource_receipt"]["caps"] == result["resource_receipt"]["attempted"] == m.design.expected()
        assert result["callback_receipts"]["attempted"] == {"packet": 4, "encode": 4}
        assert result["callback_receipts"]["physical_steps"] is None
        assert [p["end"] for p in result["packets"]] == [5, 10, 15, 20]
        assert not result["scientific_pass"] and not result["native_execution_certified"]
    assert [x["horizon"] for x in values[0]["blocks"]] == [1, 3]
    for i in (1, 2):
        paired = m.compare(values[0], values[i])
        assert paired["matched_measured_axes"] and paired["matched_hook_shape_multiset"]
        assert paired["physical_meter_match"] is None and paired["FLOPs"] is None
    assert m.compare(values[1], values[2])["matched_measured_axes"]
    assert values[1]["blocks"][0]["horizon"] != values[2]["blocks"][0]["horizon"]


def test_early_stop_no_padding_or_extra_encoding():
    class Stop(Session):
        def packet(self, start, raw):
            return dict(raw=raw[:2].copy(), rgb_bytes=b"", end=start + 2, stopped=True, episode_id=self.episode_id)
    r, events = run("reverse_signal_order", Stop())
    assert r["status"] == "native_stopped"
    assert len(r["packets"]) == 1 and r["packets"][0]["end"] == 2
    assert r["callback_receipts"]["attempted"] == {"packet": 1}
    assert r["callback_receipts"]["completed"] == {"packet": 1}
    assert not any(e["kind"] == "actual_context" for e in events)


def test_encode_failure_does_not_refund_or_replan():
    class Bad(Session):
        def encode(self, end, rgb):
            raise RuntimeError("fixture encoding fault")
    r, events = run("reverse_signal_order", Bad())
    assert r["status"] == "failed"
    assert r["callback_receipts"]["attempted"] == {"packet": 1, "encode": 1}
    assert r["callback_receipts"]["completed"] == {"packet": 1}
    assert len(r["blocks"]) == 1
    assert sum(e["kind"] == "callback_failure" for e in events) == 1
    good, _ = run("fixed_order")
    assert m.compare(good, r)["matched_measured_axes"] is None


def test_failed_replay_attempt_is_not_refunded():
    class Broken(fixture.RamBackend):
        def replay(self, *args):
            raise RuntimeError("replay failure")
    r, _ = run("reverse_signal_order", backend=Broken)
    assert r["status"] == "failed"
    assert r["resource_receipt"]["attempted"]["replay_calls"] == 1
    assert r["resource_receipt"]["completed"]["replay_calls"] == 0
    assert not r["packets"]


def test_unsupported_reverse_decision_stops_after_paid_signals(monkeypatch):
    monkeypatch.setattr(m.core, "choose", lambda *args: {
        "continue_expanding": True, "program_type": "reconfiguration",
        "extension_type": "reconfiguration", "horizon": 3})
    result, _ = run("reverse_signal_order")
    assert result["status"] == "failed"
    assert result["resource_receipt"]["attempted"]["signal_compile_calls"] == 2
    assert result["resource_receipt"]["attempted"]["program_compile_calls"] == 0
    assert not result["packets"] and not result["blocks"]


def test_tampered_budget_or_decision_rejected():
    a, _ = run("signal_order")
    b, _ = run("reverse_signal_order")
    changed = deepcopy(b)
    changed["blocks"][0]["horizon"] = a["blocks"][0]["horizon"]
    with pytest.raises(ValueError):
        m.compare(a, changed)
    changed = deepcopy(b)
    changed["resource_receipt"]["caps"]["predict_API"] += 1
    with pytest.raises(ValueError):
        m.compare(a, changed)


def test_six_slot_setup_failure_denominator_and_cleanup():
    roots = [setup()[0], setup()[0]]
    store = fixture.Events()
    closed = []
    def factory(slot, root, policy):
        if slot == 2:
            raise RuntimeError("setup")
        return Session(), setup()[1], lambda: closed.append(slot)
    result = m.execute_roster(roots, factory, lambda task, sink: fixture.RamBackend(task, sink), store, permit_ram=True)
    assert result["fixed_denominator"] == 6
    assert [(r["slot"], r["root"], r["policy"]) for r in result["rows"]] == m.ROSTER
    assert result["rows"][2]["status"] == "setup_failed"
    assert closed == [0, 1, 3, 4, 5]
    assert not result["scientific_pass"]


def test_implicit_fixture_backend_denied():
    root, prior = setup()
    r = m.execute_slot(root, "reverse_signal_order", prior, Session(),
                       lambda task, store: fixture.RamBackend(task, store), fixture.Events())
    assert r["status"] == "failed" and not r["packets"]


@pytest.mark.parametrize("policy", ["fixed_order", "signal_order"])
def test_original_arms_match_existing_feedback_seam(policy):
    old, _ = original_fixture.run(policy, Session())
    new, _ = run(policy)
    assert new["blocks"] == old["blocks"]
    assert new["packets"] == old["packets"]
    assert new["resource_receipt"]["attempted"] == old["resource_receipt"]["attempted"]
    assert new["resource_receipt"]["completed"] == old["resource_receipt"]["completed"]


def test_wrong_ack_or_encoder_frame_stops_slot():
    class BadAck(Session):
        def packet(self, start, raw):
            result = super().packet(start, raw)
            result["raw"][0, 0] += np.float32(0.01)
            return result
    r, _ = run("signal_order", BadAck())
    assert r["status"] == "failed" and not r["packets"]
    assert r["callback_receipts"]["completed"] == {"packet": 1}
    class BadFrame(Session):
        def encode(self, end, rgb):
            result = super().encode(end, rgb)
            result["input_sha256"] = "0" * 64
            return result
    r, events = run("reverse_signal_order", BadFrame())
    assert r["status"] == "failed" and len(r["packets"]) == 1
    assert r["callback_receipts"]["completed"] == {"packet": 1, "encode": 1}
    assert not any(e["kind"] == "actual_context" for e in events)
