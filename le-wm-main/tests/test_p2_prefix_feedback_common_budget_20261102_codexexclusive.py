import numpy as np
import pytest

from experiments.spectral_reachability import p2_prefix_feedback_common_budget_20261102_codexexclusive as m


def fixture_callbacks(fail_at=None, leak_state=False):
    captures, used = [], []
    base = np.zeros((2, 10), dtype="<f4")
    def start(arm, action, gradient, meter):
        meter.call("start_transition", lambda: None)
        meter.call("start_transition", lambda: None)
        if gradient:
            meter.call("start_gradient", lambda: None)
        return float(np.square(action - .2).sum()), (np.full((10, 2), -.1, np.float32) if gradient else None)
    def capture(arm, index, prefix):
        captures.append((arm, index, prefix.copy()))
        history = np.stack((base[1], m.normalization.normalized_history(prefix)[0])).astype("<f4")
        images = np.zeros((3, 224, 224, 3), np.uint8)
        images[2] = index + (0 if arm == m.ARMS[0] else 32)
        value = {"arm": arm, "candidate_index": index, "root_sha256": "root", "task_id": "task",
                 "prefix_hex": prefix.tobytes().hex(), "rgb_history": images,
                 "action_history": history, "capture_receipt_sha256": "a" * 64,
                 "terminated": fail_at == (arm, index), "truncated": False}
        if leak_state:
            value["privileged_state"] = [1]
        return value
    def encode(rgb, history):
        return {"layout": "states_with_between_state_actions_v1",
                "latent_history": np.full((3, 192), float(rgb[2, 0, 0, 0]), np.float32).tolist(),
                "action_history": history.tolist()}
    def suffix(arm, context, action, gradient, meter):
        used.append((arm, context["latent_history"][0][0]))
        meter.call("suffix_transition", lambda: None)
        if gradient:
            meter.call("suffix_gradient", lambda: None)
        return float(np.square(action[5:] - .1).sum()), (np.full((10, 2), -.1, np.float32) if gradient else None)
    return base, captures, used, start, capture, encode, suffix


def test_full_16_candidates_each_arm_uses_own_feedback_and_fixed_budget():
    base, captures, used, start, capture, encode, suffix = fixture_callbacks()
    root_rgb = [m.sha(np.zeros((224, 224, 3), np.uint8).tobytes())] * 2
    bank = [np.full((10, 2), i / 100, np.float32).tobytes() for i in range(16)]
    events = []
    result = m.run_case(bank, "root", "task", base, root_rgb, evaluate_start=start, capture=capture,
                        encode=encode, evaluate_suffix=suffix, emit=events.append)
    assert result["costs"]["attempted"] == result["costs"]["returned"] == m.LIMITS
    assert len(captures) == 32 and len(used) == 288
    for i, raw in enumerate(bank):
        conditional, joint = captures[2*i:2*i+2]
        assert conditional[0] == m.ARMS[0] and joint[0] == m.ARMS[1]
        assert conditional[2].tobytes() == raw[:40]
        assert joint[2].tobytes() != raw[:40]
        assert (m.ARMS[0], float(i)) in used and (m.ARMS[1], float(i+32)) in used
    assert len([e for e in events if e["kind"] == "feedback_bound"]) == 32
    assert all(e["elapsed_ns"] >= 0 for e in events if e["kind"] == "resource_finish")
    assert result["physical_adapter_authenticated"] is False and result["equal_total_compute"] is False


def test_stop_fails_closed_and_preserves_no_false_complete():
    base, captures, _, start, capture, encode, suffix = fixture_callbacks(fail_at=(m.ARMS[0], 2))
    root_rgb = [m.sha(np.zeros((224, 224, 3), np.uint8).tobytes())] * 2
    events = []
    bank = [np.zeros((10, 2), np.float32).tobytes()] * 16
    with pytest.raises(ValueError, match="native stop"):
        m.run_case(bank, "root", "task", base, root_rgb, evaluate_start=start, capture=capture,
                   encode=encode, evaluate_suffix=suffix, emit=events.append)
    assert len(captures) == 5
    assert len([e for e in events if e["kind"] == "feedback_bound"]) == 4
    assert events[-1]["kind"] == "root_aborted" and events[-1]["candidate_denominator"] == 16
    assert events[-1]["winners"] is None


def test_privileged_state_and_wrong_history_rejected():
    base, _, _, _, capture, _, _ = fixture_callbacks(leak_state=True)
    root_rgb = [m.sha(np.zeros((224, 224, 3), np.uint8).tobytes())] * 2
    prefix = np.zeros((5, 2), np.float32)
    with pytest.raises(ValueError, match="privileged"):
        m.validate_capture(capture(m.ARMS[0], 0, prefix), m.ARMS[0], 0, prefix, "root", "task", base, root_rgb)
    _, _, _, _, capture, _, _ = fixture_callbacks()
    value = capture(m.ARMS[0], 0, prefix)
    value["action_history"][1, 0] += 1
    with pytest.raises(ValueError, match="history"):
        m.validate_capture(value, m.ARMS[0], 0, prefix, "root", "task", base, root_rgb)
    value = capture(m.ARMS[0], 0, prefix)
    value["rgb_history"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="initial RGB"):
        m.validate_capture(value, m.ARMS[0], 0, prefix, "root", "task", base, root_rgb)


def test_budget_rejects_extra_and_cpu_model_adapter():
    meter = m.Meter(lambda e: None)
    for _ in range(m.LIMITS["feedback_encode"]):
        meter.call("feedback_encode", lambda: None)
    with pytest.raises(ValueError, match="resource cap"):
        meter.call("feedback_encode", lambda: pytest.fail("called beyond cap"))
    with pytest.raises(ValueError, match="CUDA"):
        m.native_evaluators({}, [], lambda *x: None, "cpu")
