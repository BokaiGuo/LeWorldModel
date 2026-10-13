import numpy as np

from experiments.run_rgb_local_effect_adapter import terminal_pulse_fallback_action


def test_terminal_pulse_fallback_reverses_tangent_and_keeps_inward_component():
    # Agent is below the block, so block-agent points toward the contact
    # normal.  The prior pulse has a positive tangent component.
    agent = np.array([0.0, 0.0], dtype=np.float32)
    block = np.array([0.0, 10.0], dtype=np.float32)
    previous = np.array([-0.04, 0.03], dtype=np.float32)
    fallback = terminal_pulse_fallback_action(previous, agent, block, cap=0.06)

    normal = (block - agent) / np.linalg.norm(block - agent)
    tangent = np.array([-normal[1], normal[0]], dtype=np.float32)
    assert np.linalg.norm(fallback) <= 0.06 + 1e-6
    assert float(np.dot(fallback, normal)) >= 0.019
    assert float(np.dot(fallback, tangent)) < 0.0


def test_terminal_pulse_fallback_is_zero_safe_for_zero_previous_tangent():
    agent = np.array([0.0, 0.0], dtype=np.float32)
    block = np.array([0.0, 10.0], dtype=np.float32)
    previous = np.array([0.0, 0.04], dtype=np.float32)
    fallback = terminal_pulse_fallback_action(previous, agent, block, cap=0.06)
    assert np.all(np.isfinite(fallback))
    assert float(np.linalg.norm(fallback)) <= 0.06 + 1e-6
    assert float(fallback[1]) >= 0.019
