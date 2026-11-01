import json

import numpy as np

from experiments.spectral_reachability import codex_p1_adam_feedback_plan_20261030_main as plan


def test_fixed_inputs_and_bounded_plan():
    entries = plan.inputs()
    result = json.loads((plan.base.ROOT / "results/codex_p1_adam_feedback_plan_20261030_main_RUN01.json").read_text())
    assert len(entries) == len(result["rows"]) == 24
    assert result["model_before"] == result["model_after"] == plan.base.MODEL_HASH
    for (source, physical, raw, rgb, frame), row in zip(entries, result["rows"], strict=True):
        assert row["index"] == source["index"] == physical["index"]
        assert row["first_packet_hex"] == raw[0].tobytes().hex()
        assert row["midpoint_pixel_sha256"] == frame["pixel_sha256"]
        packet = np.frombuffer(bytes.fromhex(row["second_packet_hex"]), dtype="<f4")
        assert packet.shape == (10,) and np.isfinite(packet).all() and (np.abs(packet) <= 1).all()
        assert row["chosen"] == ("optimized" if row["optimized_score"] < row["baseline_score"] else "original")
        if row["chosen"] == "original":
            assert row["second_packet_hex"] == raw[1].tobytes().hex()
