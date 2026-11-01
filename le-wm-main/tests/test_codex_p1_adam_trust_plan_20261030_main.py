import json

import numpy as np

from experiments.spectral_reachability import codex_p1_adam_trust_plan_20261030_main as trust


def test_fixed_roster_action_and_objective():
    inputs = trust.source.inputs()
    saved = json.loads((trust.base.ROOT / f"results/{trust.NAME}_RUN01.json").read_text())
    assert len(inputs) == len(saved["rows"]) == 24
    assert saved["model_before"] == saved["model_after"] == trust.base.MODEL_HASH
    for (original, physical, raw, _, frame), row in zip(inputs, saved["rows"], strict=True):
        assert original["index"] == physical["index"] == row["index"]
        assert row["first_packet_hex"] == raw[0].tobytes().hex()
        assert row["midpoint_pixel_sha256"] == frame["pixel_sha256"]
        packet = np.frombuffer(bytes.fromhex(row["second_packet_hex"]), dtype="<f4")
        assert packet.shape == (10,) and np.isfinite(packet).all() and (abs(packet) <= 1).all()
        assert np.isclose(row["optimized_score"], row["optimized_goal_mse"] +
                          trust.LAMBDA * row["optimized_action_mse"], rtol=1e-6)
        assert row["chosen"] == ("optimized" if row["optimized_score"] < row["baseline_score"] else "original")
