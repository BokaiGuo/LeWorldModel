"""DINOv2 encoder backed by the repository's offline local shim."""

import os
from pathlib import Path

import torch
import torch.nn as nn


class DinoV2Encoder(nn.Module):
    def __init__(self, name: str, feature_key: str):
        super().__init__()
        if name != "dinov2_vits14" or feature_key != "x_norm_patchtokens":
            raise ValueError("compat runtime supports only dinov2_vits14 patch tokens")
        source = Path(os.environ.get("LEWM_DINOV2_SOURCE", "/home/gbk/code/LeWorldModel/le-wm-main/external/local_dinov2"))
        self.base_model = torch.hub.load(str(source), name, source="local")
        self.emb_dim = int(self.base_model.num_features)
        self.patch_size = int(self.base_model.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model.forward_features(x)["x_norm_patchtokens"]


__all__ = ["DinoV2Encoder"]

