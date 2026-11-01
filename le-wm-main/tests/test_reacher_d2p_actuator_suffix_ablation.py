import numpy as np
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_actuator_suffix_ablation import render_frame, suffix_effect_metrics


def test_render_frame_preserves_flipped_rgb_values_with_tensor_compatible_strides():
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)[::-1]

    class Renderer:
        def render(self):
            return pixels

    frame = render_frame(Renderer())
    np.testing.assert_array_equal(frame, pixels)
    assert frame.flags.c_contiguous
    assert torch.as_tensor(frame).shape == (4, 6, 3)


def test_suffix_intervention_separates_passive_motion_from_control_effect():
    def method(passive, optimized):
        return {"prefix_then_zero_suffix": {"final_qpos_distance": passive, "observed_rgb_goal_cost": passive * 2},
            "optimized": {"final_qpos_distance": optimized, "observed_rgb_goal_cost": optimized * 2}}
    result = suffix_effect_metrics({"dispatch": method(1.5, .2), "same_bank_greedy": method(.8, .3)})
    assert result["dispatch_minus_greedy_suffix_improvement"] == pytest.approx(.8)
    assert result["dispatch_passive_minus_greedy_passive"] > 0
    assert result["dispatch_optimized_minus_greedy_optimized"] < 0


def test_identical_actions_do_not_invent_a_reconfiguration_effect():
    method = {"prefix_then_zero_suffix": {"final_qpos_distance": 1., "observed_rgb_goal_cost": 2.},
        "optimized": {"final_qpos_distance": .3, "observed_rgb_goal_cost": .2}}
    result = suffix_effect_metrics({"dispatch": method, "same_bank_greedy": method})
    assert result["dispatch_minus_greedy_suffix_improvement"] == 0
    assert result["dispatch_passive_minus_greedy_passive"] == 0
