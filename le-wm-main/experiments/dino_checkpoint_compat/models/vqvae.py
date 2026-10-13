"""Decoder shells required only by safe checkpoint object deserialization."""

import torch
from torch import nn


class VQVAE(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class Quantize(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class Decoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ResBlock(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


__all__ = ["VQVAE", "Quantize", "Decoder", "ResBlock"]
