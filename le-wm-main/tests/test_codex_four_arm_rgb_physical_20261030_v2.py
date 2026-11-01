import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_rgb_physical_20261030_v2 as m


def test_first_packet_preserves_order_and_values():
    actions = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2) / 30
    np.testing.assert_array_equal(m.first_packet(actions), actions[0].numpy())


@pytest.mark.parametrize("actions", [
    torch.zeros(5, 2),
    torch.zeros(1, 3, 5, 2),
    torch.full((3, 5, 2), float("nan")),
    torch.full((3, 5, 2), 1.01),
])
def test_first_packet_rejects_bad_plans(actions):
    with pytest.raises((ValueError, AssertionError)):
        m.first_packet(actions)
