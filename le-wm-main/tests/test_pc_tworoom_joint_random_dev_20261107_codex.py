import numpy as np
import pytest
import torch

from experiments import pc_tworoom_joint_random_dev_20261107_codex as run


def test_packet_normalization_and_gradient():
    blocks = torch.arange(10, dtype=torch.float32).reshape(5, 2).requires_grad_()
    packed = run.packets(blocks, torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    assert packed.shape == (1, 5, 10)
    np.testing.assert_allclose(packed.detach().numpy()[0, 0], [-0.5, -0.25] * 5)
    packed.sum().backward()
    np.testing.assert_allclose(blocks.grad.numpy(), np.tile([2.5, 1.25], (5, 1)))


def test_saved_random_adam_and_selection():
    if not run.POOL.exists():
        pytest.skip("sealed development artifact not present")
    with np.load(run.POOL, allow_pickle=False) as pool:
        for root in range(10, 50):
            x = pool[f"r{root}_adam_actions"]
            gradients = pool[f"r{root}_adam_gradients"]
            r = pool[f"r{root}_random_actions"]
            chosen = pool[f"r{root}_chosen"]
            assert x.shape == r.shape == (9, 5, 2)
            assert np.max(np.abs(x)) <= 1 and np.max(np.abs(r)) <= 1
            rng = np.random.default_rng(491000 + root)
            expected = np.clip(x[0][None] + rng.normal(0, 0.15, (8, 5, 2)), -1, 1).astype(np.float32)
            np.testing.assert_array_equal(r[0], x[0])
            np.testing.assert_array_equal(r[1:], expected)
            params = torch.tensor(x[0], requires_grad=True)
            opt = torch.optim.Adam([params], lr=0.05)
            for i in range(8):
                params.grad = torch.tensor(gradients[i])
                opt.step()
                with torch.no_grad():
                    params.clamp_(-1, 1)
                np.testing.assert_allclose(params.detach().numpy(), x[i + 1], atol=3e-7, rtol=1e-6)
            for idx, name in ((1, "adam"), (2, "random")):
                best = np.argmin(pool[f"r{root}_{name}_scores"])
                np.testing.assert_array_equal(chosen[idx], np.repeat(pool[f"r{root}_{name}_actions"][best], 5, axis=0))
