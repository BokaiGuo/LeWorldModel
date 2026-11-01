import numpy as np
import pytest

from experiments.spectral_reachability import p1_v10_cp5_rgb_suffix_design_20261101_parallel as m


def test_actual_v8_index16_is_frozen_into_v10():
    value = m.build()
    source = value["v8_source"]
    assert source["selected_index"] == 16
    assert source["label"] == "rgb_delayed_x23_neg04"
    assert source["prefix_actions_sha256"] == (
        "0db61c3546f87f17bdf69efa4b55e6a8e9678ae7b4627c1c3cebd731d58bdb36"
    )
    assert len(source["prefix_actions"]) == 5
    assert len(source["old_tail_actions"]) == 10
    assert value["P1_admission"] is False
    assert value["execution_authorized"] is False


def test_matched_arms_have_exactly_equal_full_rollout_budget():
    value = m.build()
    arms = value["matched_design"]["arms"]
    assert set(arms) == {"v10_contact_normal", "matched_unconstrained"}
    assert arms["v10_contact_normal"]["budget"] == arms["matched_unconstrained"]["budget"]
    budget = arms["v10_contact_normal"]["budget"]
    assert budget["iterations"] == 5
    assert budget["samples"] == 64
    assert budget["variants"] == 5
    assert budget["full_suffix_steps"] == 10
    assert budget["predict_samples"] == 3200


def test_support_normal_and_first_three_contact_constraint():
    normal = m.support_normal([10.0, 10.0], [20.0, 10.0])
    assert np.allclose(normal, [1.0, 0.0])
    object_pixels = np.asarray(
        [[10.0, 9.0], [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]]
    )
    centers = np.asarray([[4.0, 10.0], [4.5, 10.0], [5.0, 10.0], [20.0, 20.0]])
    result = m.contact_hold(centers, object_pixels, normal)
    assert result["valid"] is True
    assert len(result["steps"]) == 3
    assert min(row["normal_alignment"] for row in result["steps"]) >= 0.8
    bad = centers.copy()
    bad[1] = [10.0, 4.0]
    assert m.contact_hold(bad, object_pixels, normal)["valid"] is False


def test_projection_is_in_pixel_motion_space_and_keeps_prefix_velocity():
    normal = np.asarray([1.0, 0.0])
    tail = np.zeros((10, 2), np.float32)
    previous = np.asarray([0.0, 1.0], np.float32)
    projected = m.project_early_motion(tail, normal, previous)
    centers = m.predict_pusher_centers([0.0, 0.0], projected, previous)
    displacements = np.diff(
        np.concatenate((np.zeros((1, 2)), centers[: m.CONTACT_STEPS]), axis=0),
        axis=0,
    )
    assert np.all(displacements @ normal >= m.MIN_FORWARD_PIXELS - 1e-6)
    tangential = displacements - (displacements @ normal)[:, None] * normal
    assert np.all(np.linalg.norm(tangential, axis=1) <= m.MAX_TANGENTIAL_PIXELS + 1e-6)
    without_carry = m.predict_pusher_centers([0.0, 0.0], projected, [0.0, 0.0])
    assert not np.allclose(centers[0], without_carry[0])


def test_robust_full_rollout_uses_all_variants_and_rejects_bad_contact():
    costs = np.asarray([
        [1.0, 1.1, 0.9, 1.05, 0.95],
        [0.8, 0.9, 0.7, 0.85, 0.75],
    ])
    contact_valid = np.ones_like(costs, dtype=bool)
    contact_valid[1, 4] = False
    scores = m.robust_full_rollout_scores(costs, contact_valid)
    assert scores[0] == pytest.approx(costs[0].mean() + np.ptp(costs[0]))
    assert np.isinf(scores[1])
    with pytest.raises(ValueError):
        m.robust_full_rollout_scores(costs[:, :4], contact_valid)
    with pytest.raises(ValueError):
        m.robust_full_rollout_scores(costs, [True, True])


def test_design_does_not_bind_v9():
    value = m.build()
    assert all("v9" not in path.lower() for path in value["bindings"])
    assert value["cp5_information_boundary"]["post_cp5_outcome_used"] is False
    geometry = value["task_support_geometry"]
    assert "actually executed fifth prefix action" in geometry["pusher_motion_model"]
    assert value["robust_full_rollout"]["prefix_velocity_reset"] == "prohibited"
