import numpy as np
import pytest

from experiments.spectral_reachability import p1_devroot_blind_payload_20261101_hypatia as b


def test_collect_source_frames_and_actual_actions():
    packets, provenance, _ = b.collect()
    assert len(packets) == len(provenance) == 2
    assert all(set(p) == {"rgb_history", "raw_action_history", "goal_rgb"} for p in packets)
    assert all(np.array_equal(p["raw_action_history"], b.encoder.fixed_raw()) for p in packets)
    assert all(row["goal_frame"].endswith("construction_20.png") for row in provenance)


def test_reject_duplicate_missing_and_bad_action():
    event = {"kind": "observation", "slot": 0, "phase": "replay0", "step": 0}
    with pytest.raises(ValueError, match="unique"):
        b.one([event, event], kind="observation", slot=0, phase="replay0", step=0)
    with pytest.raises(ValueError, match="unique"):
        b.one([], kind="observation", slot=0, phase="replay0", step=0)
    events = [{"kind": "command", "slot": 0, "phase": "replay0", "step": i + 1,
               "raw_float32_bits": np.zeros(2, np.float32).tobytes().hex()} for i in range(10)]
    with pytest.raises(ValueError, match="action bits"):
        b.warmup_actions(events, 0)


def test_encoder_rejects_extra_privileged_field():
    packet = {"rgb_history": np.zeros((3, 224, 224, 3), np.uint8),
              "raw_action_history": b.encoder.fixed_raw(),
              "goal_rgb": np.zeros((224, 224, 3), np.uint8),
              "state": np.zeros(7)}
    with pytest.raises(ValueError):
        b.encoder.check_packet(packet)
