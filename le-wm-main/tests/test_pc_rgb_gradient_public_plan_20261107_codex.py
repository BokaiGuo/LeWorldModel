import numpy as np
import pytest
import torch

from experiments.pc_rgb_gradient_public_plan_20261107_codex import normalize_actions, validate_public


def package(tmp_path, **extra):
    path = tmp_path / "public.npz"
    values = dict(episodes=np.array([1]), current=np.zeros((1, 8, 8, 3), np.uint8),
                  goal=np.zeros((1, 8, 8, 3), np.uint8), nominal=np.zeros((1, 25, 2), np.float32),
                  action_mean=np.array([0., 0.]), action_scale=np.array([1., 1.]))
    values.update(extra)
    np.savez(path, **values)
    return np.load(path, allow_pickle=False)


def test_public_whitelist_rejects_state_and_outcome(tmp_path):
    with package(tmp_path) as valid:
        validate_public(valid)
    with package(tmp_path, success=np.array([True])) as poisoned:
        with pytest.raises(ValueError):
            validate_public(poisoned)


def test_normalization_preserves_gradient():
    actions = torch.ones((25, 2), requires_grad=True)
    result = normalize_actions(actions, np.array([.5, .2]), np.array([2., 4.]))
    result.sum().backward()
    torch.testing.assert_close(actions.grad, torch.tensor([.5, .25]).expand_as(actions))
