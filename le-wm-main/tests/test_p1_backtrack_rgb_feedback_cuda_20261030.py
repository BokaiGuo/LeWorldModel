import numpy as np
import pytest

from experiments.spectral_reachability.p1_backtrack_rgb_feedback_cuda_20261030 import feedback_contexts


def test_contexts_use_actual_root_and_observed_checkpoints():
    source = np.stack([np.full(192, i, np.float32) for i in range(3)])
    actual = np.stack([np.full(192, i, np.float32) for i in range(10, 14)])
    contexts = feedback_contexts(source, actual)
    assert contexts[:, :, 0].tolist() == [[1, 10, 11], [10, 11, 12]]


def test_rejects_invalid_latents():
    with pytest.raises(ValueError, match="context"):
        feedback_contexts(np.zeros((3, 192)), np.full((4, 192), np.nan))
