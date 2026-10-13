import numpy as np

from experiments.run_pusht_counterfactual_observability_decomposition import (
    _corr,
    bootstrap_corr,
    color_mask_features,
    pixel_features,
)


def test_correlation_helpers_detect_monotone_association():
    x = np.arange(1.0, 9.0)
    y = 2.0 * x + 1.0
    assert _corr(x, y) > 0.99
    assert _corr(x, y, spearman=True) > 0.99
    result = bootstrap_corr(x, y, seed=9, draws=100)
    assert result["estimate"] > 0.99
    assert result["finite_bootstrap_draws"] > 0


def test_correlation_helpers_keep_constant_inputs_nonfinite():
    result = bootstrap_corr(np.ones(8), np.arange(8.0), seed=9, draws=100)
    assert not np.isfinite(result["estimate"])


def test_pixel_capacity_control_has_expected_shape():
    images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
    features = pixel_features(images, size=8)
    assert features.shape == (3, 8 * 8 * 3)
    assert np.isfinite(features).all()


def test_color_mask_capacity_control_tracks_renderer_block_not_goal():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[60:100, 80:120] = [143, 163, 184]  # movable block
    image[120:160, 140:180] = [144, 238, 144]  # goal
    image[180:190, 40:50] = [65, 105, 225]  # agent
    features = color_mask_features([image])
    assert features.shape == (1, 9)
    assert np.isfinite(features).all()
    # The centroid must follow the grey-blue block, not the green goal.
    assert 95.0 < features[0, 0] < 105.0
    assert 75.0 < features[0, 1] < 85.0


def test_color_mask_capacity_control_is_finite_on_blank_images():
    images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
    features = color_mask_features(images)
    assert features.shape == (2, 9)
    assert np.isfinite(features).all()
