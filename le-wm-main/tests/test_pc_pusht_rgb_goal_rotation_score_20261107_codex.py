import numpy as np
import pytest

from experiments.pc_pusht_rgb_goal_rotation_score_20261107_codex import goal_variants


def test_abstain_and_input_contract():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    variants, info = goal_variants(image)
    assert len(variants) == 1 and info["abstained"]
    np.testing.assert_array_equal(variants[0], image)
    with pytest.raises(ValueError):
        goal_variants(image.astype(np.float32))


def test_rgb_variants_keep_original_and_are_deterministic():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    image[80:120, 90:110] = [120, 137, 154]
    image[105:120, 75:125] = [120, 137, 154]
    original = image.copy()
    variants, info = goal_variants(image)
    assert not info["abstained"] and len(variants) == 3
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(variants[0], original)
    assert not np.array_equal(variants[1], variants[2])
    variants2, _ = goal_variants(image)
    for left, right in zip(variants, variants2):
        np.testing.assert_array_equal(left, right)
