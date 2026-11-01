from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_cp10_rgb_refresh_verify_20261101_exclusive as verifier


def events_for(role, end=15):
    events = []
    for step in range(1, end + 1):
        events.extend([{"kind": "branch_dispatch", "role": role, "step": step, "command": [0, 0]},
                       {"kind": "branch_returned", "role": role, "step": step, "command": [0, 0],
                        "terminated": step == end and end < 15, "truncated": False}])
        if step in (5, 10) and step < end:
            events.append({"kind": "decision_sealed", "role": role, "boundary": step})
    return [dict(e, index=i) for i, e in enumerate(events)]


def test_dispatch_requires_seal_same_episode_and_no_post_stop():
    events = events_for("selected_rgb_refresh")
    verifier.check_order(events, "selected_rgb_refresh", 15)
    broken = deepcopy(events)
    broken = [e for e in broken if e["kind"] != "decision_sealed" or e["boundary"] != 10]
    with pytest.raises(ValueError):
        verifier.check_order(broken, "selected_rgb_refresh", 15)
    broken = deepcopy(events)
    next(e for e in broken if e["kind"] == "decision_sealed")["role"] = "another_episode"
    with pytest.raises(ValueError):
        verifier.check_order(broken, "selected_rgb_refresh", 15)
    verifier.check_order(events_for("z", 7), "z", 7)
    broken = events_for("z", 7) + [{"kind": "branch_dispatch", "role": "z", "step": 8, "command": [0, 0]}]
    with pytest.raises(ValueError):
        verifier.check_order(broken, "z", 7)


def test_neural_attempts_and_returns_must_pair_and_obey_budget():
    events = [{"kind": "neural_attempt", "role": "a", "resource": "predict", "context": {"step": 5}},
              {"kind": "neural_return", "role": "a", "resource": "predict", "context": {"step": 5}}]
    counts = {"actual_rgb_encode": 0, "action_encoder": 0, "predict": 1}
    verifier.check_neural(events, "a", counts, counts)
    with pytest.raises(ValueError):
        verifier.check_neural(events[:1], "a", counts, counts)
    broken = deepcopy(events)
    broken[1]["context"] = {"step": 10}
    with pytest.raises(ValueError):
        verifier.check_neural(broken, "a", counts, counts)
    with pytest.raises(ValueError):
        verifier.check_neural(events * 7, "a", dict(counts, predict=7), dict(counts, predict=7))


def test_early_stop_records_no_decision_or_missing_action_as_observed():
    assert verifier.unobserved(15, 15) == 0
    assert verifier.unobserved(15, 10) == 5
    with pytest.raises(ValueError):
        verifier.unobserved(15, 16)


def test_neural_expected_counts_follow_boundaries_not_horizon_flag():
    assert verifier.expected_neural(5) == {"actual_rgb_encode": 0, "action_encoder": 0, "predict": 0}
    assert verifier.expected_neural(10) == {"actual_rgb_encode": 1, "action_encoder": 4, "predict": 4}
    assert verifier.expected_neural(15) == {"actual_rgb_encode": 2, "action_encoder": 6, "predict": 6}


def test_capture_censor_is_preserved_and_not_filled():
    verifier.check_frame_count(7, 7, "censored_out_of_domain", 7)
    verifier.check_frame_count(7, 8, "native_stopped", None)
    with pytest.raises(ValueError):
        verifier.check_frame_count(7, 7, "native_stopped", None)
    with pytest.raises(ValueError):
        verifier.check_frame_count(7, 6, "censored_out_of_domain", 7)
    verifier.check_frame_count(0, 0, "native_stopped_warmup", None)


def test_cp10_pair_requires_actions_and_each_identity_field():
    point = {k: "same" for k in verifier.spec.IDENTITY}
    row = {"observations": [point] * 11, "executed_raw": [[0, 0]] * 10}
    reference = verifier.cp10_pair(row)
    assert verifier.cp10_pair(row) == reference
    changed = deepcopy(row)
    changed["observations"][10]["pixel_sha256"] = "different"
    assert verifier.cp10_pair(changed) != reference
    changed = deepcopy(row)
    changed["executed_raw"][6] = [1, 0]
    assert verifier.cp10_pair(changed) != reference
