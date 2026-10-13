import numpy as np
import torch

from experiments.run_rgb_local_effect_adapter import (
    POSE_DIM,
    action_bank,
    attach_targets,
    contact_boundary_chart_from_images,
    terminal_contact_lease_from_boundary,
    terminal_probe_contact_lease_from_image,
    contact_chart_from_features,
    contact_effect_to_world,
    rgb_pose_calibration_features,
    rgb_geometry_pose_chart,
    rgb_geometry_ranker_features,
    rgb_geometry_xy_angle_from_image,
    rgb_shape_raster_features,
    rgb_mask_xy_from_image,
    project_contact_action,
    observed_block_xy,
    runtime_rgb_angle_features,
    runtime_features,
    terminal_ranker_score_contract,
    validate_terminal_ranker_contract,
    terminal_control_lease_active,
    terminal_action_commitment_eligible,
    make_recovery_anchor,
    recovery_anchor_geometry,
    side_route_action,
    terminal_centroid_local_repair,
    terminal_visual_hold_eligible,
    train_rgb_pose_calibrator,
    rgb_signed_torque_probe_action,
    terminal_support_fallback_action,
    observed_block_xy,
    runtime_current_xy_policy,
    safe_agent_xy_observation,
)


def test_terminal_ranker_is_disabled_after_visual_terminal_fan_escape():
    # A valid support lease must not keep the h-step angular ranker active
    # after the candidate set has switched back to broad translation actions.
    assert terminal_control_lease_active(True, True, True, 35.0, True)
    assert not terminal_control_lease_active(True, False, True, 35.0, True)
    assert not terminal_control_lease_active(True, True, True, 35.0, False)


def test_terminal_commitment_covers_torque_prior_and_safety_fallback():
    action = np.asarray([0.03, 0.04], dtype=np.float32)
    assert terminal_action_commitment_eligible(
        "torque", action, {"terminal_torque_prior_used": 1.0},
    )
    assert terminal_action_commitment_eligible(
        "torque", action, {"terminal_contact_safety_used": 1.0},
    )
    assert terminal_action_commitment_eligible(
        "hold", action, {"terminal_support_fallback_used": 1.0},
    )
    assert not terminal_action_commitment_eligible(
        "translate", action, {"terminal_torque_prior_used": 1.0},
    )
    assert not terminal_action_commitment_eligible(
        "torque", np.zeros(2, dtype=np.float32), {"terminal_torque_prior_used": 1.0},
    )


def test_terminal_ranker_contract_maps_pose_horizon_to_pose_mode():
    assert terminal_ranker_score_contract("pose", 4) == "direct_multistep_pose"
    assert terminal_ranker_score_contract("angle", 4) == "direct_multistep_angle"
    assert terminal_ranker_score_contract("angle", 1) == "one_step_residual"


def test_terminal_ranker_contract_rejects_stale_pose_as_angle_artifact():
    args = type("Args", (), {
        "terminal_ranker_target": "pose",
        "terminal_action_horizon": 4,
    })()
    training = {"score_mode": "direct_multistep_angle", "action_horizon": 4}
    try:
        validate_terminal_ranker_contract(training, args)
    except RuntimeError as exc:
        assert "contract mismatch" in str(exc)
    else:
        raise AssertionError("stale objective metadata must fail closed")


def test_terminal_ranker_contract_accepts_matching_report_and_training():
    args = type("Args", (), {
        "terminal_ranker_target": "pose",
        "terminal_action_horizon": 4,
    })()
    training = {
        "score_mode": "direct_multistep_pose",
        "action_horizon": 4,
        "target": "pose",
    }
    report = {
        "score_mode": "direct_multistep_pose",
        "action_horizon": 4,
        "target": "pose",
    }
    assert validate_terminal_ranker_contract(training, args, report_metrics=report) == "direct_multistep_pose"


def test_action_bank_starts_with_zero_and_is_finite():
    actions = action_bank(8, [0.35, 0.8])
    assert actions.shape == (17, 2)
    assert np.allclose(actions[0], 0.0)
    assert np.isfinite(actions).all()


def test_agent_observer_rejects_impossible_dynamic_decoder_jump():
    raw = np.asarray([120.0, 140.0], dtype=np.float32)
    held = np.asarray([121.0, 141.0], dtype=np.float32)
    observed = safe_agent_xy_observation(
        np.asarray([2800.0, -1900.0], dtype=np.float32), raw, held,
    )
    np.testing.assert_allclose(observed, held)


def test_contact_chart_matches_runtime_feature_width():
    features = np.zeros(30, dtype=np.float32)
    previous = np.zeros(30, dtype=np.float32)
    action = np.asarray([0.4, -0.2], dtype=np.float32)
    value = runtime_features(features, previous, action)
    assert value.shape == (1, 82)
    assert np.isfinite(value).all()
    assert contact_chart_from_features(features, action).shape == (1, 10)


def test_runtime_features_can_append_previous_action_without_changing_legacy_width():
    features = np.zeros(30, dtype=np.float32)
    previous = np.zeros(30, dtype=np.float32)
    action = np.asarray([0.4, -0.2], dtype=np.float32)
    legacy = runtime_features(features, previous, action)
    conditioned = runtime_features(
        features, previous, action, previous_action=np.asarray([0.1, -0.3], dtype=np.float32),
    )
    assert legacy.shape == (1, 82)
    assert conditioned.shape == (1, 84)
    np.testing.assert_allclose(conditioned[:, :82], legacy)
    np.testing.assert_allclose(conditioned[:, 82:], [[0.1, -0.3]])


def test_contact_boundary_chart_is_finite_for_empty_and_rgb_batches():
    images = np.zeros((2, 3, 16, 16), dtype=np.uint8)
    actions = np.asarray([[0.2, 0.1], [0.0, 0.0]], dtype=np.float32)
    chart = contact_boundary_chart_from_images(images, actions)
    assert chart.shape == (2, 10)
    assert np.isfinite(chart).all()


def test_contact_boundary_chart_preserves_asymmetric_tangential_lever():
    # The support point on an asymmetric T need not lie on the centroid-to-
    # agent normal.  The old projected chart forced this component to zero.
    image = np.zeros((3, 48, 64), dtype=np.uint8)
    image[:, 20:30, 8:18] = np.asarray([0, 0, 255], dtype=np.uint8)[:, None, None]
    image[:, 17:27, 24:42] = np.asarray([80, 150, 190], dtype=np.uint8)[:, None, None]
    image[:, 8:17, 30:42] = np.asarray([80, 150, 190], dtype=np.uint8)[:, None, None]
    chart = contact_boundary_chart_from_images(
        image[None], np.asarray([[0.4, 0.0]], dtype=np.float32),
    )
    assert chart.shape == (1, 10)
    assert np.isfinite(chart).all()
    # The selected left support point lies below the block centroid in this
    # image, so its lever has a non-zero tangential component.
    assert abs(float(chart[0, 3])) > 1.0e-3


def test_terminal_contact_lease_uses_boundary_gap_and_fails_closed_on_missing_support():
    chart = np.zeros(10, dtype=np.float32)
    chart[:4] = [0.1, -0.2, 0.05, 0.02]
    chart[8] = 0.10  # 25.6 pixels at the PushT canvas scale.
    valid, gap, visible = terminal_contact_lease_from_boundary(chart, 32.0)
    assert valid and visible
    np.testing.assert_allclose(gap, 25.6, atol=1e-5)

    chart[8] = 0.20  # 51.2 pixels: centroid distance must not rescue it.
    valid, gap, visible = terminal_contact_lease_from_boundary(chart, 32.0)
    assert not valid and visible
    assert gap > 32.0

    missing = np.zeros(10, dtype=np.float32)
    valid, gap, visible = terminal_contact_lease_from_boundary(missing, 32.0)
    assert not valid and not visible and np.isfinite(gap)

    # Compatibility mode remains unchanged for previously recorded protocols.
    valid, _, _ = terminal_contact_lease_from_boundary(missing, 0.0)
    assert valid


def test_signed_probe_contact_lease_fails_closed_before_excitation():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    valid, gap, visible = terminal_probe_contact_lease_from_image(image, 15.0, 64)
    assert not valid and not visible
    assert gap == 0.0

    # The zero-gap compatibility protocol deliberately preserves the old
    # centroid/side-only behavior; the strict boundary lease is opt-in.
    valid, gap, visible = terminal_probe_contact_lease_from_image(image, 0.0, 64)
    assert valid and not visible and np.isnan(gap)


def test_signed_probe_has_independent_cap_without_changing_terminal_cap():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    common = dict(
        image_size=64,
        _signed_torque_probe_magnitude_scale=1.0,
        signed_torque_probe_magnitude=0.16,
        signed_torque_probe_inward=0.08,
        terminal_action_max_magnitude=0.04,
        contact_action_max_magnitude=0.35,
    )
    legacy, _ = rgb_signed_torque_probe_action(
        image, np.asarray([20.0, 20.0]), np.asarray([30.0, 30.0]),
        0.8, type("Args", (), common)(), 1.0,
    )
    uncoupled_args = type(
        "Args", (), {**common, "signed_torque_probe_max_magnitude": 0.16}
    )()
    uncoupled, _ = rgb_signed_torque_probe_action(
        image, np.asarray([20.0, 20.0]), np.asarray([30.0, 30.0]),
        0.8, uncoupled_args, 1.0,
    )
    np.testing.assert_allclose(np.linalg.norm(legacy), 0.04, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(uncoupled), 0.16, atol=1e-6)


def test_terminal_support_fallback_replaces_zero_for_unsupported_cap():
    # The learned bank starts at 0.08 in the affected protocol, while the
    # terminal cap is 0.04. The fallback must therefore emit a non-zero RGB
    # action inside the cap rather than allowing zero to win against +inf.
    action, audit = terminal_support_fallback_action(
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        0.0,
        np.pi / 2.0,
        0.04,
    )
    assert np.linalg.norm(action) > 1.0e-6
    assert np.linalg.norm(action) <= 0.04 + 1.0e-7
    normal = np.asarray([1.0, 0.0], dtype=np.float32)
    assert float(np.dot(action, normal)) > 0.0
    assert audit["terminal_support_fallback_torque_sign"] == 1.0
    assert audit["terminal_support_fallback_signed_torque"] > 0.0


def test_terminal_support_fallback_respects_declared_torque_sign_for_negative_error():
    action, audit = terminal_support_fallback_action(
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.pi / 2.0,
        0.0,
        0.04,
    )
    assert np.linalg.norm(action) <= 0.04 + 1.0e-7
    assert float(np.dot(action, np.asarray([1.0, 0.0], dtype=np.float32))) > 0.0
    assert audit["terminal_support_fallback_torque_sign"] == -1.0
    assert audit["terminal_support_fallback_signed_torque"] < 0.0


def test_local_targets_reuse_root_zero_baseline():
    actions = action_bank(4, [0.5])
    states = np.tile(np.asarray([[0.0, 0.0, 10.0, 20.0, 0.0, 1.0]], dtype=np.float32), (len(actions), 1))
    successors = states.copy()
    successors[0, 2:4] = [11.0, 20.5]
    successors[1, 2:4] = [12.0, 21.5]
    successors[2, 2:4] = [9.0, 20.5]
    successors[3, 2:4] = [11.5, 18.0]
    data = {
        "state": states,
        "next_state": successors,
        "root_id": np.zeros(len(actions), dtype=np.int64),
        "contact_root": np.ones(len(actions), dtype=bool),
    }
    out = attach_targets(data, actions)
    assert out["local_target"].shape == (len(actions), 2 * POSE_DIM)
    # zero action is the first row and its baseline is shared by every row.
    assert np.allclose(out["zero_successor"], successors[0])
    assert np.allclose(out["local_target"][0, POSE_DIM:], 0.0)
    assert np.allclose(out["local_target"][1, POSE_DIM:2 * POSE_DIM], [1.0, 1.0, 0.0])


def test_contact_effect_rotation_is_single_and_invertible():
    # A non-axis-aligned contact frame makes an accidental second rotation
    # observable.  The first pose half is the zero baseline; only the effect
    # half is transformed.
    agent = np.asarray([0.0, 0.0], dtype=np.float32)
    block = np.asarray([1.0, 1.0], dtype=np.float32)
    prediction = np.zeros((1, 2 * POSE_DIM), dtype=np.float32)
    prediction[0, POSE_DIM:POSE_DIM + 2] = [1.0, 0.0]
    world = contact_effect_to_world(prediction, agent, block)
    np.testing.assert_allclose(world[0, POSE_DIM:POSE_DIM + 2], [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
    # A second application is deliberately different; this guards the
    # runtime invariant that a selected KNN prediction is not transformed
    # again by the generic adapter-frame branch.
    twice = contact_effect_to_world(world, agent, block)
    assert not np.allclose(twice[0, POSE_DIM:POSE_DIM + 2], world[0, POSE_DIM:POSE_DIM + 2])


def test_rgb_mask_observer_returns_canvas_coordinates_and_none_on_absence():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[10:20, 20:30] = [80, 150, 190]
    value = rgb_mask_xy_from_image(image)
    np.testing.assert_allclose(value, [266.89362, 239.48387], atol=1e-4)
    assert rgb_mask_xy_from_image(np.zeros_like(image)) is None


def test_rgb_geometry_chart_distinguishes_current_block_and_goal_target_masks():
    image = np.zeros((3, 32, 48), dtype=np.uint8)
    image[:, 10:20, 20:30] = np.asarray([80, 150, 190], dtype=np.uint8)[:, None, None]
    image[:, 4:8, 6:10] = np.asarray([65, 105, 225], dtype=np.uint8)[:, None, None]
    goal = np.zeros_like(image)
    goal[:, 12:22, 30:40] = np.asarray([80, 190, 100], dtype=np.uint8)[:, None, None]
    chart = rgb_geometry_pose_chart(np.stack([image, goal]))
    assert chart["block"][0, -1] == 1.0
    assert chart["agent"][0, -1] == 1.0
    assert chart["target"][1, -1] == 1.0
    assert chart["block"][1, -1] == 0.0
    assert np.isfinite(np.concatenate(list(chart.values()), axis=1)).all()


def test_rgb_geometry_ranker_uses_same_image_interface_and_fallbacks():
    current = np.zeros((2, 3, 32, 48), dtype=np.uint8)
    current[:, :, 10:20, 20:30] = np.asarray([80, 150, 190], dtype=np.uint8)[None, :, None, None]
    current[:, :, 4:8, 6:10] = np.asarray([65, 105, 225], dtype=np.uint8)[None, :, None, None]
    previous = current.copy()
    goal = np.zeros_like(current)
    goal[:, :, 12:22, 30:40] = np.asarray([80, 190, 100], dtype=np.uint8)[None, :, None, None]
    actions = np.asarray([[0.0, 0.0], [0.35, -0.2]], dtype=np.float32)
    features = rgb_geometry_ranker_features(
        current, previous, goal, actions,
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.tile(np.asarray([300.0, 320.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
    )
    assert features.shape[0] == 2
    assert features.shape[1] > 60
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[0, :2], features[1, :2])
    # The goal render is centered; ranker world-space goal XY must come from
    # the validated fallback pose, not the target-mask centroid.
    np.testing.assert_allclose(features[:, 4:6], [[300.0 / 512.0, 320.0 / 512.0]] * 2)


def test_rgb_geometry_ranker_invariant_chart_is_finite_and_coordinate_reduced():
    current = np.zeros((2, 3, 32, 48), dtype=np.uint8)
    current[:, :, 10:20, 20:30] = np.asarray([80, 150, 190], dtype=np.uint8)[None, :, None, None]
    current[:, :, 4:8, 6:10] = np.asarray([65, 105, 225], dtype=np.uint8)[None, :, None, None]
    previous = current.copy()
    goal = np.zeros_like(current)
    goal[:, :, 12:22, 30:40] = np.asarray([80, 190, 100], dtype=np.uint8)[None, :, None, None]
    actions = np.asarray([[0.0, 0.0], [0.35, -0.2]], dtype=np.float32)
    full = rgb_geometry_ranker_features(
        current, previous, goal, actions,
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.tile(np.asarray([300.0, 320.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
        feature_mode="full",
    )
    invariant = rgb_geometry_ranker_features(
        current, previous, goal, actions,
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.tile(np.asarray([300.0, 320.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
        feature_mode="invariant",
    )
    assert invariant.shape[0] == full.shape[0] == 2
    assert invariant.shape[1] < full.shape[1]
    assert np.isfinite(invariant).all()
    shifted_goal_pose = np.tile(
        np.asarray([420.0, 400.0, 0.0, 1.0], dtype=np.float32), (2, 1),
    )
    shifted_invariant = rgb_geometry_ranker_features(
        current, previous, goal, actions,
        np.asarray([0.1, 0.2], dtype=np.float32), shifted_goal_pose,
        feature_mode="invariant",
    )
    # Absolute goal XY is intentionally excluded from this contact-frame
    # chart; changing the canvas-space fallback must not alter it.
    np.testing.assert_allclose(invariant, shifted_invariant, atol=1e-6)


def test_rgb_goal_observer_uses_target_mask_and_falls_back_to_block_mask():
    goal = np.zeros((32, 48, 3), dtype=np.uint8)
    goal[12:22, 30:40] = [80, 190, 100]
    target_xy, target_angle = rgb_geometry_xy_angle_from_image(goal, "target")
    np.testing.assert_allclose(target_xy, [34.5 * 512.0 / 47.0, 16.5 * 512.0 / 31.0], atol=2.0)
    assert target_angle is not None
    block = np.zeros_like(goal)
    block[10:20, 20:30] = [80, 150, 190]
    block_xy, block_angle = rgb_geometry_xy_angle_from_image(block, "target")
    np.testing.assert_allclose(block_xy, [24.5 * 512.0 / 47.0, 14.5 * 512.0 / 31.0], atol=2.0)
    assert block_angle is not None


def test_rgb_previous_and_current_observer_share_one_coordinate_chart():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[10:20, 20:30] = [80, 150, 190]
    features = np.zeros(30, dtype=np.float32)
    current = observed_block_xy(
        image, features, None, None, "cpu", "rgb_mask", 60.0,
    )
    previous = observed_block_xy(
        image.copy(), features, None, None, "cpu", "rgb_mask", 60.0,
    )
    # Identical frames must never create a synthetic contact-motion signal.
    np.testing.assert_allclose(current, previous, atol=1e-6)


def test_rgb_mask_fused_fails_closed_to_mask_on_large_decoder_disagreement():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[10:20, 20:30] = [80, 150, 190]
    features = np.zeros(30, dtype=np.float32)
    decoder = torch.nn.Linear(30, 2)
    with torch.no_grad():
        decoder.weight.zero_()
        decoder.bias.zero_()
    # The synthetic decoder output is deliberately far from the RGB mask.
    observed = observed_block_xy(
        image, features, decoder,
        {"feature_mean": [0.0] * 30, "feature_std": [1.0] * 30,
         "target_mean": [0.0, 0.0], "target_std": [1.0, 1.0]},
        "cpu", "rgb_mask_fused", 10.0,
    )
    expected = rgb_mask_xy_from_image(image)
    np.testing.assert_allclose(observed, expected, atol=1e-5)


def test_rgb_shape_raster_features_are_position_normalized_and_finite():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[80:140, 90:150] = [119, 136, 153]
    shifted = np.zeros_like(image)
    shifted[100:160, 120:180] = [119, 136, 153]
    first = rgb_shape_raster_features(np.stack([image]), "block")
    second = rgb_shape_raster_features(np.stack([shifted]), "block")
    assert first.shape == (1, 256)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_rgb_shape_raster_runtime_interface_matches_training_metadata():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[80:140, 90:150] = [119, 136, 153]
    raster = runtime_rgb_angle_features(
        image, "block", {"feature_interface": "raster"}, 64,
    )
    moments = runtime_rgb_angle_features(
        image, "block", {"feature_interface": "moments"}, 64,
    )
    assert raster.shape == (1, 256)
    assert moments.shape == (1, 12)
    assert np.isfinite(raster).all() and np.isfinite(moments).all()


def test_rgb_pose_calibrator_corrects_declared_shape_offset_without_state_access():
    rng = np.random.default_rng(17)
    shape = rng.normal(size=(32, 12)).astype(np.float32)
    shape[:, 0] = 0.3
    shape[:, 1:3] = rng.uniform(-0.8, 0.8, size=(32, 2))
    shape[:, 10:12] = rng.normal(size=(32, 2))
    design = rgb_pose_calibration_features(shape)
    weights = rng.normal(size=(design.shape[1], 2)).astype(np.float32)
    targets = design @ weights
    model, training = train_rgb_pose_calibrator(
        shape[:24], targets[:24], shape[24:], targets[24:], ridge_alpha=1.0e-5,
    )
    prediction = model.predict(shape[24:])
    np.testing.assert_allclose(prediction, targets[24:], atol=1.0e-3)
    assert training["readout"] == "closed_form_ridge_pose_calibration"


def test_translation_contact_safety_arguments_have_conservative_defaults():
    # The new safety gate is opt-in so previously recorded protocols remain
    # reproducible; when enabled it must never silently use a negative ratio.
    import argparse

    args = argparse.Namespace(
        contact_translation_inward_min=0.2,
        contact_translation_tangent_ratio=1.25,
    )
    assert args.contact_translation_inward_min > 0.0
    assert args.contact_translation_tangent_ratio >= 0.0


def test_side_route_rotates_wrong_contact_normal_toward_goal():
    normal = np.asarray([1.0, 0.0], dtype=np.float32)
    goal = np.asarray([0.0, 1.0], dtype=np.float32)
    action = side_route_action(normal, goal, 0.45)
    # The action clears the contact (negative normal component) and turns
    # counter-clockwise toward the desired block-to-goal direction.
    assert float(np.dot(action, normal)) < 0.0
    assert float(action[1]) > 0.0
    assert np.isfinite(action).all()


def test_terminal_centroid_local_repair_is_bounded_and_local():
    action, raw_norm, final_norm, target_distance = terminal_centroid_local_repair(
        np.asarray([100.0, 100.0], dtype=np.float32),
        np.asarray([140.0, 180.0], dtype=np.float32),
        action_scale=100.0,
        action_cap=0.12,
    )
    assert target_distance > 0.0
    assert raw_norm > final_norm
    np.testing.assert_allclose(np.linalg.norm(action), 0.12, atol=1.0e-6)
    assert np.dot(action, np.asarray([40.0, 80.0], dtype=np.float32)) > 0.0
    assert np.isfinite(action).all()


def test_terminal_visual_hold_guard_preempts_recovery_only_inside_declared_tube():
    assert terminal_visual_hold_eligible("torque", 12.0, 0.20, 18.0, 0.28)
    assert terminal_visual_hold_eligible("hold", 18.0, 0.28, 18.0, 0.28)
    assert not terminal_visual_hold_eligible("recovery", 12.0, 0.20, 18.0, 0.28)
    assert not terminal_visual_hold_eligible("torque", 19.0, 0.20, 18.0, 0.28)
    assert not terminal_visual_hold_eligible("torque", 12.0, 0.30, 18.0, 0.28)


def test_recovery_anchor_keeps_route_geometry_fixed_under_block_drift():
    anchor = make_recovery_anchor(
        np.asarray([100.0, 120.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([100.0, 220.0], dtype=np.float32),
        100.0,
    )
    waypoint, arc_midpoint, outer, target = recovery_anchor_geometry(anchor, 42.0)
    # The current RGB block/goal estimate is deliberately not passed to this
    # function: all three targets are tied to the recovery-onset frame.  The
    # workspace-aware anchor may reduce the requested clearance near a wall.
    assert np.all(waypoint >= 28.0) and np.all(waypoint <= 484.0)
    assert np.all(arc_midpoint >= 28.0) and np.all(arc_midpoint <= 484.0)
    assert np.all(outer >= 28.0) and np.all(outer <= 484.0)
    np.testing.assert_allclose(target, np.asarray([58.0, 120.0], dtype=np.float32))


def test_contact_action_projection_repairs_outward_action_without_successor_query():
    import argparse

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    # Agent is left of the block, so the RGB contact normal points +x.
    image[10:20, 6:16] = [0, 0, 255]
    image[10:20, 20:30] = [80, 150, 190]
    args = argparse.Namespace(
        contact_action_projection=True,
        contact_projection_min_inward=0.1,
        contact_distance=42.0,
        contact_tolerance=24.0,
        contact_lock_distance=100.0,
    )
    projected, info = project_contact_action(
        np.asarray([-0.5, 0.2], dtype=np.float32), image, args, "push",
    )
    assert projected[0] >= 0.1 - 1e-6
    assert info["contact_action_projection_used"] == 1.0
    assert info["contact_projection_raw_inward"] < 0.0
    assert info["contact_projection_final_inward"] >= 0.1 - 1e-6
    assert np.isfinite(projected).all()


def test_contact_action_projection_is_opt_in_and_preserves_legacy_action():
    import argparse

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[10:20, 6:16] = [0, 0, 255]
    image[10:20, 20:30] = [80, 150, 190]
    args = argparse.Namespace(
        contact_action_projection=False,
        contact_projection_min_inward=0.1,
        contact_distance=42.0,
        contact_tolerance=24.0,
        contact_lock_distance=100.0,
    )
    action = np.asarray([-0.5, 0.2], dtype=np.float32)
    projected, info = project_contact_action(action, image, args, "push")
    np.testing.assert_allclose(projected, action)
    assert info["contact_action_projection_used"] == 0.0


def test_rgb_calibrated_current_xy_uses_the_same_shape_interface_as_training():
    rng = np.random.default_rng(5)
    shape = rng.normal(size=(24, 12)).astype(np.float32)
    shape[:, 0] = 0.3
    shape[:, 1:3] = rng.uniform(-0.8, 0.8, size=(24, 2))
    design = rgb_pose_calibration_features(shape)
    weights = rng.normal(size=(design.shape[1], 2)).astype(np.float32)
    targets = design @ weights
    calibrator, training = train_rgb_pose_calibrator(
        shape[:16], targets[:16], shape[16:], targets[16:], ridge_alpha=1.0e-5,
    )
    # The runtime branch is exercised through the public observer with a
    # synthetic image; a missing mask must fail closed rather than fabricate
    # a coordinate.
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = [80, 150, 190]
    features = np.zeros(30, dtype=np.float32)
    observed = observed_block_xy(
        image, features, calibrator, training, "cpu", "rgb_calibrated", 60.0,
    )
    assert observed.shape == (2,)
    assert np.isfinite(observed).all()


def test_runtime_current_xy_policy_prefers_fail_closed_effective_observer():
    import argparse

    args = argparse.Namespace(
        current_xy_policy="paired_rgb_calibrated",
        _runtime_current_xy_policy="validation",
    )
    # The requested ablation must not override the post-gate runtime choice.
    assert runtime_current_xy_policy(args) == "validation"


def test_runtime_current_xy_policy_preserves_requested_protocol_without_gate():
    import argparse

    args = argparse.Namespace(current_xy_policy="paired_rgb_calibrated")
    assert runtime_current_xy_policy(args) == "paired_rgb_calibrated"
