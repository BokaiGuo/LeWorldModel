import cv2
import numpy as np

from experiments.run_rgb_local_effect_adapter import (
    rgb_temporal_orientation_increment,
    temporal_rgb_angle_observation,
)


def _t_shape(angle: float = 0.0) -> np.ndarray:
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    color = (64, 128, 180)  # satisfies the declared blue-block RGB mask
    cv2.rectangle(image, (90, 90), (134, 106), color, -1)
    cv2.rectangle(image, (108, 90), (122, 145), color, -1)
    if angle:
        matrix = cv2.getRotationMatrix2D((112, 112), -np.degrees(angle), 1.0)
        image = cv2.warpAffine(
            image, matrix, (224, 224), borderValue=(255, 255, 255),
        )
    return image


def test_temporal_registration_recovers_small_signed_increment():
    result = rgb_temporal_orientation_increment(_t_shape(0.0), _t_shape(0.1))
    assert result["valid"] == 1.0
    assert result["delta"] > 0.05
    assert result["delta"] < 0.16


def test_temporal_observer_fails_closed_on_missing_mask():
    blank = np.full((224, 224, 3), 255, dtype=np.uint8)
    state = {}
    selected, audit = temporal_rgb_angle_observation(
        blank, blank, 0.4, 0.4, state,
    )
    assert np.isfinite(selected)
    assert audit["temporal_angle_update_used"] == 0.0
    assert audit["temporal_angle_delta"] == 0.0

