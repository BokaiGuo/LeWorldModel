import numpy as np
import torch

from experiments.spectral_reachability.audit_reacher_goal_readout_alignment import _decoded_qpos_goal_cost, _embedding, _packets, _rgb_goal_cost


def test_goal_readout_packet_bank_is_seeded_bounded_and_keeps_zero_control():
    first = _packets(9, packets=8)
    second = _packets(9, packets=8)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 5, 2)
    assert np.count_nonzero(first[0]) == 0
    assert np.all(first >= -1.0) and np.all(first <= 1.0)


def test_embedding_materializes_negative_stride_renderer_frame():
    class Preprocessor:
        def _prepare_info(self, info):
            pixels = info["pixels"]
            assert pixels.flags.c_contiguous
            return {"pixels": torch.from_numpy(pixels)}

    class Model:
        def encode(self, info):
            return {"emb": info["pixels"].float().mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)}

    # Simulate the vertically flipped view returned by the MuJoCo renderer.
    image = np.zeros((5, 7, 3), dtype=np.uint8)[::-1]
    embedding = _embedding(Model(), Preprocessor(), image, torch.device("cpu"))
    assert tuple(embedding.shape) == (1,)


def test_rgb_goal_cost_is_zero_for_an_exact_render_and_positive_after_change():
    goal = np.zeros((3, 4, 3), dtype=np.uint8)
    changed = goal.copy()
    changed[0, 0, 1] = 255
    assert _rgb_goal_cost(goal, goal) == 0.0
    assert _rgb_goal_cost(changed, goal) > 0.0


def test_decoded_qpos_goal_cost_uses_circular_joint_distance():
    assert _decoded_qpos_goal_cost(np.asarray([-np.pi + 0.01, 0.0]), np.asarray([np.pi - 0.01, 0.0])) < 0.03
