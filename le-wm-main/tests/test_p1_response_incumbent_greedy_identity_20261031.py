"""Check whether the two named controls are distinct actions on the fixed panel."""
import json

from experiments.spectral_reachability import p1_response_midpoint_controls_20261031 as audit


def test_incumbent_is_exact_greedy_replay():
    assert all(audit.sha(path) == digest for path, digest in audit.SOURCES)
    new, old, _, new_plan, old_plan, _, _, online, midpoint_plan = [
        json.loads(path.read_text()) for path, _ in audit.SOURCES
    ]
    candidates = [row for row in new["anchors"] if row["role"] == "incumbent"]
    assert len(candidates) == 8
    events = {row["anchor"]: row for row in online["online_events"]}
    for row in candidates:
        i, pair = row["index"], row["logical_pair"]
        baseline = old["anchors"][3 * pair + 2]
        proposal = new_plan["rows"][i]
        reference = old_plan["rows"][3 * pair + 2]
        midpoint = midpoint_plan["rows"][3 * pair + 2]
        assert baseline["role"] == midpoint["role"] == reference["role"] == "greedy"
        assert row["prefix5_float32_hex"] == baseline["prefix5_float32_hex"]
        assert row["checkpoint"] == baseline["checkpoint"]
        assert proposal["initial_f32_sha256"] == reference["initial_f32_sha256"]
        assert proposal["actions_sha256"] == reference["selected_action_sha256"]
        assert events[i]["midpoint_pixel_sha256"] == midpoint["midpoint_pixel_sha256"]
        assert events[i]["second_packet_hex"] == midpoint["second_packet_hex"]
        assert events[i]["baseline_score"] == midpoint["baseline_score"]
        assert events[i]["optimized_score"] == midpoint["optimized_score"]
