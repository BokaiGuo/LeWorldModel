import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_source_rgb_contract import (
    _pixel_error,
    source_marker_position,
)


def test_source_marker_uses_recorded_xy_and_preserves_model_z_coordinate():
    marker = source_marker_position(np.asarray([9.0, 8.0, 7.0]), np.asarray([0.1, -0.2]))
    np.testing.assert_allclose(marker, [0.1, -0.2, 7.0])


def test_source_marker_rejects_incompatible_or_nonfinite_target_coordinates():
    with pytest.raises(ValueError, match="source target marker"):
        source_marker_position(np.asarray([0.0, 0.0, 0.0]), np.asarray([0.1]))
    with pytest.raises(ValueError, match="source target marker"):
        source_marker_position(np.asarray([0.0, 0.0, 0.0]), np.asarray([np.nan, 0.1]))


def test_pixel_error_is_zero_only_for_an_exact_source_render():
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    changed = source.copy()
    changed[0, 0, 1] = 7
    assert _pixel_error(source, source)["mean_abs_pixel_error"] == 0.0
    assert _pixel_error(changed, source)["changed_pixel_fraction"] > 0.0
