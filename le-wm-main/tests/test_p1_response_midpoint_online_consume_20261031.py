import copy

import pytest

from experiments.spectral_reachability import p1_response_midpoint_online_consume_20261031 as consumer


def fixture():
    item = {"index": 1, "midpoint": {"pixel_sha256": "pixels"},
            "raw10_float32_hex": "ab" * 40 + "cd" * 40}
    expected = {"anchor": 1, "midpoint_pixel_sha256": "pixels", "second_packet_hex": "cd" * 40,
                "baseline_score": 1.0, "optimized_score": 0.9, "optimizer_seconds": 0.1,
                "calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20}}
    events = [{"index": i, "kind": "fresh_observation",
               "receipt": {"phase": "probe", "step": 5}} for i in (2, 3, 4)]
    events += [dict(index=5, kind="online_midpoint_plan", **expected),
               {"index": 6, "kind": "returned_transition", "phase": "probe", "step": 6}]
    return item, expected, events


def test_online_event_must_precede_sixth_step_and_follow_real_images():
    item, expected, events = fixture()
    consumer.check_events(events, item, expected)
    events[-1]["index"] = 4
    with pytest.raises(Exception, match="ordering"):
        consumer.check_events(events, item, expected)


def test_tampered_action_is_rejected_and_pre_midpoint_stop_has_no_event():
    item, expected, events = fixture()
    tampered = copy.deepcopy(events)
    tampered[-2]["second_packet_hex"] = "00" * 40
    with pytest.raises(Exception, match="identity"):
        consumer.check_events(tampered, item, expected)
    consumer.check_events([], item, None)
