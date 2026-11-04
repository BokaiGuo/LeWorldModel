import copy
import json

import pytest

from experiments.spectral_reachability import pc_ap_durable_saved_consumer_20261102_codexexclusive as consumer


@pytest.fixture
def real_arm():
    path = consumer.INPUT / "20263160_ap_observed_target.jsonl"
    if not path.exists():
        pytest.skip("saved development journal unavailable")
    events, _ = consumer.read_chain(path)
    ref = json.loads(consumer.REFERENCE.read_text())
    row = next(x for x in ref["rows"] if x["seed"] == 20263160)
    return events, (row["root_rgb_sha256"], row["goal_rgb_sha256"])


def test_saved_arm_and_mutated_step_action(real_arm):
    events, root = real_arm
    assert consumer.audit_arm(events, 20263160, "ap_observed_target", root)["status"] == "complete"
    changed = copy.deepcopy(events)
    next(x["event"] for x in changed if x.get("event", {}).get("kind") == "step_returned")["action_float32_hex"] = "00" * 8
    with pytest.raises(ValueError, match="step returned action"):
        consumer.audit_arm(changed, 20263160, "ap_observed_target", root)


@pytest.mark.parametrize("mutation", ["count", "stop", "history", "rgb", "sequence"])
def test_counterexamples_fail_closed(real_arm, mutation):
    original, root = real_arm
    events = copy.deepcopy(original)
    if mutation == "count":
        next(x for x in events if x["kind"] == "predictor_call_returned")["blocks"] = 999
    elif mutation == "stop":
        next(x["event"] for x in events if x.get("event", {}).get("kind") == "step_returned")["terminated"] = True
    elif mutation == "history":
        c = next(x for x in events if x["kind"] == "context_bound")["context"]
        c["history_action_float32_hex"][0] = "00" * 40
    elif mutation == "rgb":
        c = next(x for x in events if x["kind"] == "context_bound")["context"]
        c["root"]["sha256"] = "0" * 64
    else:
        i = next(i for i, x in enumerate(events) if x["kind"] == "evaluator_distance")
        events[i], events[i+1] = events[i+1], events[i]
    with pytest.raises(ValueError):
        consumer.audit_arm(events, 20263160, "ap_observed_target", root)


def test_chain_corruption_and_truncation(tmp_path):
    body = {"index": 0, "previous_sha256": None, "payload": {"kind": "x"}}
    body["event_sha256"] = consumer.sha(consumer.canonical(body))
    path = tmp_path / "journal"
    path.write_bytes(consumer.canonical(body) + b"\n")
    assert consumer.read_chain(path)[1]["events"] == 1
    body["payload"]["kind"] = "tampered"
    path.write_bytes(consumer.canonical(body) + b"\n")
    with pytest.raises(ValueError, match="event hash"):
        consumer.read_chain(path)
    path.write_bytes(b'{"incomplete"')
    with pytest.raises(ValueError, match="truncated"):
        consumer.read_chain(path)


def test_failure_and_cost_are_preserved():
    events = [{"kind": "predictor_call_intent", "requested_blocks": 16},
              {"kind": "predictor_call_failed", "blocks": 16, "batches": 1},
              {"kind": "bridge_event", "event": {"kind": "step_intent"}},
              {"kind": "bridge_event", "event": {"kind": "step_failed", "step": 1}},
              {"kind": "arm_failed", "message": "failed"}]
    summary = consumer.cost_summary(events)
    assert summary["observed"]["predictor_requested_blocks"] == 16
    assert summary["observed"]["predictor_failed_calls"] == 1
    assert summary["observed"]["step_intents"] == 1
    assert len(summary["failures_and_stops"]) == 2


def test_normalization_mismatch_detected(real_arm):
    import numpy as np
    events, root = real_arm
    with pytest.raises(ValueError, match="normalization"):
        consumer.audit_arm(events, 20263160, "ap_observed_target", root, (np.zeros(10), np.ones(10)))
