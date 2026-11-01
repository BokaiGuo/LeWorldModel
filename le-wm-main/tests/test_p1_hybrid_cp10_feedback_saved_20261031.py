import copy

import pytest

from experiments.spectral_reachability.p1_hybrid_cp10_feedback_saved_20261031 import verify_midpoint_gate


def fixture():
    trace = [{"step": n, "state": [n], "pixel_sha256": str(n)} for n in range(1, 6)]
    midpoint = {"state": [5], "pixel_sha256": "5"}
    frames = [{"kind": "fresh_observation", "receipt": dict(item, phase="probe", overlay_index=0)}
              for item in trace]
    verified = {"kind": "midpoint_refresh_verified", "index": 10,
                "checkpoint": midpoint, "trace": trace}
    dispatch = {"kind": "operation_dispatch_intent", "index": 11,
                "receipt": {"resource": "public_step", "phase": "probe", "step": 6}}
    return frames + [verified, dispatch], midpoint, trace


def test_feedback_gate_rejects_wrong_frame_or_late_verification():
    events, midpoint, trace = fixture()
    verify_midpoint_gate(events, midpoint, trace)
    bad = copy.deepcopy(events)
    bad[2]["receipt"]["pixel_sha256"] = "wrong"
    with pytest.raises(ValueError, match="feedback gate"):
        verify_midpoint_gate(bad, midpoint, trace)
    bad = copy.deepcopy(events)
    bad[-1]["index"] = 9
    with pytest.raises(ValueError, match="feedback gate"):
        verify_midpoint_gate(bad, midpoint, trace)
