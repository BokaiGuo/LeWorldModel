import numpy as np

from experiments.spectral_reachability.p1_dev305_perturbation_audit_20261102 import pixel_effects


def test_translation_sign_and_zero_identity():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    image[70:90, 70:90] = [120, 140, 160]
    image[140:150, 140:150] = [40, 80, 230]
    zero = pixel_effects(image, "object_x", 0)
    plus = pixel_effects(image, "object_x", 1)
    minus = pixel_effects(image, "object_x", -1)
    assert zero["changed_pixels_total"] == 0
    np.testing.assert_allclose(plus["target_centroid_delta_xy"], [4, 0])
    np.testing.assert_allclose(minus["target_centroid_delta_xy"], [-4, 0])
    assert plus["other_entity_changed_pixels"] == 0
