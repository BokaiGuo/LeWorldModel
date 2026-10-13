"""Deterministic preprocessing for the public DINO-WM temporal path."""

from __future__ import annotations

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


class _DinoTransform:
    def __init__(self, img_size: int = 196):
        self.img_size = int(img_size)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
        return resize(frames, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC, antialias=True)


def default_transform(img_size: int = 196):
    # The checkpoint positional embedding is 3 x 14 x 14; the requested
    # img_size is accepted for API compatibility but the model contract is
    # deliberately fixed at 196 pixels.
    del img_size
    return _DinoTransform(196)

