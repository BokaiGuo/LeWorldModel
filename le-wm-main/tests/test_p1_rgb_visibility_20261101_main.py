import numpy as np
import pytest

from experiments.spectral_reachability.p1_rgb_visibility_20261101_main import inspect


def image():
    pixels = np.full((224, 224, 3), 255, np.uint8)
    pixels[100:120, 100:120] = [143, 163, 184]
    pixels[50:60, 50:60] = [65, 105, 225]
    return pixels


def test_interior_separated_masks():
    result = inspect(image())
    assert result["eligible"]
    assert result["object"]["pixels"] == 400
    assert result["pusher"]["pixels"] == 100


def test_object_at_border_rejected():
    pixels = image()
    pixels[100:120, 220:] = [143, 163, 184]
    assert not inspect(pixels)["eligible"]


def test_empty_or_wrong_dtype_rejected():
    assert not inspect(np.full((224,224,3),255,np.uint8))["eligible"]
    with pytest.raises(ValueError):
        inspect(image().astype(float))
