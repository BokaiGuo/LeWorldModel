import numpy as np

from experiments.pc_prefix_novel_actual_encoding_20261105_codex import native_world_pixels
from stable_worldmodel.wrapper.default import AddPixelsWrapper


def test_native_resize_matches_world_pixels_wrapper():
    import gymnasium as gym

    class Dummy(gym.Env):
        def __init__(self, frame):
            self.frame = frame

        def render(self):
            return self.frame

    rng = np.random.default_rng(13)
    frame = rng.integers(0, 256, size=(513, 511, 3), dtype=np.uint8)
    wrapped = AddPixelsWrapper(Dummy(frame), pixels_shape=(224, 224))
    actual, _ = wrapped._get_pixels()
    assert np.array_equal(native_world_pixels(frame), actual["pixels"])
