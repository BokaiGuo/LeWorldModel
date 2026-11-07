"""Versioned HF DINOv2 adapter with the pinned Meta positional convention.

This is an implementation repair, not a fitted alignment or a new set of
weights. The historical local_dinov2 shim is deliberately left untouched.
Reference: dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8.
"""
import math

import torch
from torch import nn
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings


class SourceMatchedEmbeddings(Dinov2Embeddings):
    """Keep the existing parameters; change only positional interpolation."""

    def __init__(self, original):
        nn.Module.__init__(self)
        if not isinstance(original, Dinov2Embeddings):
            raise TypeError("expected HuggingFace DINOv2 embeddings")
        if isinstance(original, SourceMatchedEmbeddings):
            raise ValueError("source convention is already installed")
        if original.config.hidden_dropout_prob != 0:
            raise ValueError("only the zero-dropout released encoder is certified")
        self.config = original.config
        self.cls_token = original.cls_token
        if original.use_mask_token:
            self.mask_token = original.mask_token
        self.position_embeddings = original.position_embeddings
        self.patch_embeddings = original.patch_embeddings
        self.dropout = original.dropout
        self.patch_size = original.patch_size
        self.use_mask_token = original.use_mask_token
        self.train(original.training)

    def interpolate_pos_encoding(self, embeddings, height, width):
        count = self.position_embeddings.shape[1] - 1
        if embeddings.shape[1] - 1 == count and height == width:
            return self.position_embeddings
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("source position table must be square")
        target = (height // self.patch_size, width // self.patch_size)
        if min(target) < 1:
            raise ValueError("image smaller than a patch")
        # Meta's default vits14 uses scale_factor with a 0.1 offset, not size.
        position = self.position_embeddings.float()
        patches = position[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patches = nn.functional.interpolate(
            patches, mode="bicubic", align_corners=False, antialias=False,
            scale_factor=tuple(float(n + .1) / side for n in target),
        )
        if tuple(patches.shape[-2:]) != target:
            raise ValueError("source positional grid does not match image")
        patches = patches.permute(0, 2, 3, 1).reshape(1, -1, embeddings.shape[-1])
        return torch.cat((position[:, :1], patches), dim=1).to(embeddings.dtype)


def install_source_interpolation(backbone):
    """Explicit opt-in; preserve every parameter object, value and state key."""
    before = dict(backbone.named_parameters())
    backbone.embeddings = SourceMatchedEmbeddings(backbone.embeddings)
    after = dict(backbone.named_parameters())
    if before.keys() != after.keys() or any(after[k] is not p for k, p in before.items()):
        raise AssertionError("interpolation repair must not replace model weights")
    return backbone
