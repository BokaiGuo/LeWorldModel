import numpy as np

from experiments.audit_pusht_rgb_orientation_observability import (
    binary_metrics,
    format_metric,
    has_binary_support,
    preference_label,
    rgb_history_features,
    wrap_angle,
)


def test_wrap_angle_preserves_small_signed_difference():
    assert abs(float(wrap_angle(2.0 * np.pi + 0.3)) - 0.3) < 1.0e-8


def test_preference_label_requires_effect_margin_and_uses_offline_angle_effect():
    # +y has the stronger action response after common zero-pulse drift.
    label, margin = preference_label(0.0, 0.03, 0.38, -0.10, minimum_margin=0.05)
    assert label == 1
    assert margin > 0.05
    tied, _ = preference_label(0.0, 0.03, 0.10, -0.04, minimum_margin=0.05)
    assert tied is None


def test_rgb_history_features_are_deterministic_and_state_free():
    previous = np.zeros((12, 10, 3), dtype=np.uint8)
    current = previous.copy()
    current[2:5, 3:7, 1] = 255
    features_a = rgb_history_features(previous, current, size=8)
    features_b = rgb_history_features(previous, current, size=8)
    assert features_a.shape == (8 * 8 * 3 * 3,)
    assert np.array_equal(features_a, features_b)
    assert np.isfinite(features_a).all()


def test_binary_metrics_exposes_chance_control():
    metrics = binary_metrics(np.asarray([0.9, 0.2, 0.8, 0.1]), np.asarray([1, 0, 1, 0]))
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["majority_accuracy"] == 0.5


def test_format_metric_preserves_undefined_balanced_accuracy():
    assert format_metric(None) == "n/a"
    assert format_metric(0.625) == "0.625"


def test_binary_support_requires_both_classes():
    assert has_binary_support(np.asarray([0, 1, 1]))
    assert not has_binary_support(np.asarray([0, 0, 0]))
