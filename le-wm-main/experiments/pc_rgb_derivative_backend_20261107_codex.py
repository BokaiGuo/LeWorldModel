"""Scoped backend policy for RGB model higher-order action derivatives.

PyTorch TF32 backend settings are process-global: callers must serialize this
context against concurrent model operations that depend on different settings.
This policy is numerical infrastructure, not a scientific gate assertion.
"""
from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def precise_action_derivatives():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
