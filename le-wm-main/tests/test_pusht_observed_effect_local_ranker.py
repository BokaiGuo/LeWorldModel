import numpy as np

from experiments import run_pusht_observed_effect_local_ranker as ranker


def test_rgb_chart_descriptor_and_effect_use_only_rendered_frames():
    previous = np.zeros((12, 12, 3), dtype=np.uint8)
    current = previous.copy()
    current[3:8, 4:9, 0] = 255
    successor = current.copy()
    successor[4:9, 4:9, 1] = 255

    context = ranker.rgb_history_descriptor(previous, current, image_size=6)
    effect = ranker.rgb_observed_effect(current, successor, image_size=6)

    assert context.shape == (6 * 6 * 3 * 3,)
    assert effect.shape == (6 * 6 * 3,)
    assert np.isfinite(context).all()
    assert np.isfinite(effect).all()


def test_local_chart_ranks_declared_action_from_rgb_effect_only():
    contexts = np.asarray([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0]])
    plus_effects = np.asarray([[1.0, 0.0], [0.9, 0.0], [0.0, 0.0]])
    minus_effects = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.2, 0.0]])
    chart = ranker.fit_local_probe_chart(contexts, plus_effects, minus_effects)

    prediction = ranker.predict_local_pair_effects(chart, np.asarray([0.05, 0.0]), neighbors=2)

    assert prediction["preferred_action"] == "plus"
    assert prediction["plus_response"] > prediction["minus_response"]
    assert prediction["support_distance"] < 0.1
    assert prediction["runtime_privileged_state_access"] is False
    assert prediction["runtime_successor_access"] is False


def test_rgb_context_normalization_does_not_turn_subpixel_noise_into_false_ood_support():
    # RGB descriptors live in [0, 1].  A static background has near-zero
    # empirical train variance, but a 0.01 intensity change must not become a
    # 1,000-sigma out-of-support event merely because that pixel was constant
    # in the small chart.
    chart = ranker.fit_local_probe_chart(
        np.asarray([[0.0, 0.0], [1.0e-8, 0.0]]),
        np.asarray([[1.0], [1.0]]),
        np.asarray([[0.0], [0.0]]),
    )
    prediction = ranker.predict_local_pair_effects(chart, np.asarray([0.01, 0.0]), neighbors=1)

    assert prediction["support_distance"] < 0.25


def test_gate_fails_closed_when_calibration_cannot_form_finite_support_bound():
    calibration = ranker.calibrate_probe_gate(
        support_distances=np.asarray([0.1, 0.2]),
        observed_contrasts=np.asarray([0.4, 0.5]),
        alpha=0.10,
        minimum_roots=3,
    )

    decision = ranker.gate_local_prediction(
        {"support_distance": 0.01, "predicted_contrast": 1.0}, calibration
    )

    assert calibration["status"] == "insufficient_calibration"
    assert decision["release"] is False
    assert decision["reason"] == "insufficient_calibration"


def test_gate_requires_both_support_and_probe_contrast():
    calibration = ranker.calibrate_probe_gate(
        support_distances=np.asarray([0.10, 0.15, 0.20, 0.22, 0.25]),
        observed_contrasts=np.asarray([0.20, 0.30, 0.40, 0.50, 0.60]),
        alpha=0.20,
        minimum_roots=3,
    )
    assert calibration["status"] == "calibrated"

    released = ranker.gate_local_prediction(
        {"support_distance": 0.11, "predicted_contrast": 0.70}, calibration
    )
    unsupported = ranker.gate_local_prediction(
        {"support_distance": 9.0, "predicted_contrast": 0.70}, calibration
    )
    weak = ranker.gate_local_prediction(
        {"support_distance": 0.11, "predicted_contrast": 0.01}, calibration
    )

    assert released["release"] is True
    assert unsupported == {"release": False, "reason": "insufficient_support"}
    assert weak == {"release": False, "reason": "insufficient_probe_contrast"}


def test_calibration_uses_the_same_smoothed_chart_contrast_seen_at_runtime():
    chart = ranker.fit_local_probe_chart(
        np.asarray([[0.0], [2.0]]),
        np.asarray([[0.0], [2.0]]),
        np.asarray([[0.0], [0.0]]),
    )
    # The raw held probe has a response of 100, but the local chart at context
    # 1 predicts the mean response 1.  A deployment gate must calibrate 1,
    # not the inaccessible post-hoc raw held effect 100.
    support, contrast = ranker._calibration_predictions(
        chart,
        np.asarray([[1.0]]),
        np.asarray([[100.0]]),
        np.asarray([[0.0]]),
        neighbors=2,
    )

    assert support.shape == (1,)
    assert contrast.tolist() == [1.0]


def test_offline_privileged_label_is_kept_out_of_runtime_chart_inputs():
    label, margin = ranker.offline_orientation_preference(
        before_angle=0.0,
        zero_after_angle=0.02,
        plus_after_angle=0.40,
        minus_after_angle=-0.05,
        minimum_margin=0.05,
    )
    boundary = ranker.runtime_interface_boundary()

    assert label == 1
    assert margin > 0.05
    assert boundary["runtime_input"] == "RGB history and declared candidate actions only"
    assert boundary["offline_privileged_labels_only"] is True
