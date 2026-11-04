import numpy as np
import pytest
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import DATA
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge


@pytest.mark.parametrize("root", (1880, 1881, 1882, 1883))
def test_bridge_replays_archive_frames_without_exposing_state(root):
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).numpy()
    actions = torch.load(DATA / "actions.pth", map_location="cpu", weights_only=True).numpy()
    images = torch.load(DATA / "obses" / f"episode_{root:03d}.pth",
                        map_location="cpu", weights_only=True).numpy()
    with NativeRGBBridge(states[root, 0]) as bridge:
        assert np.array_equal(bridge.initial_rgb, images[0])
        for start in (0, 5):
            image = bridge.step(np.clip(actions[root, start:start + 5], -1, 1))
            assert bridge.step_count == start + 5
            assert np.array_equal(image, images[start + 5])
        assert not hasattr(bridge, "state")


def test_bad_packet_fails_before_any_physical_step():
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).numpy()
    with NativeRGBBridge(states[1880, 0]) as bridge:
        for invalid in (np.zeros((0, 2)), np.zeros((6, 2)), np.full((1, 2), np.nan),
                        np.array([[1.01, 0.0]])):
            with pytest.raises(ValueError):
                bridge.step(invalid)
        assert bridge.step_count == 0
