import pytest
import torch

from experiments.pc_rgb_derivative_backend_20261107_codex import precise_action_derivatives


def test_backend_settings_restored_after_exception():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(RuntimeError, match="failure"):
            with precise_action_derivatives():
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                raise RuntimeError("failure")
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def test_backend_nested_restores():
    with precise_action_derivatives():
        with precise_action_derivatives():
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
