"""Compatibility loader for published LeWM checkpoints.

The published ViT weights use the HuggingFace Transformers 4.x parameter
names, while the installed stable-pretraining 1.7 constructor exposes the
equivalent attention/MLP modules with the newer compact names.  Loading a
checkpoint without this explicit remap fails loudly; silently using a
randomly initialized encoder would invalidate every learned-model result.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import stable_worldmodel as swm
from hydra.utils import instantiate


def _resolve_local(name: str, cache_dir: str | None = None) -> tuple[Path, Path]:
    root = Path(swm.data.utils.get_cache_dir(cache_dir)) / "checkpoints"
    path = Path(name)
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        weights = path / "weights.pt"
        config = path / "config.json"
    elif path.suffix == ".pt":
        weights = path
        config = path.parent / "config.json"
    else:
        raise FileNotFoundError(f"local LeWM checkpoint not found: {path}")
    if not weights.exists() or not config.exists():
        raise FileNotFoundError(f"checkpoint requires {weights} and {config}")
    return weights, config


def remap_legacy_vit_keys(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """Map legacy HuggingFace ViT names to the current stable-pretraining names."""
    replacements = (
        (".encoder.layer.", ".layers."),
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
    )
    out: dict[str, torch.Tensor] = {}
    changed = 0
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            if old in new_key:
                new_key = new_key.replace(old, new)
        if new_key != key:
            changed += 1
        if new_key in out:
            raise ValueError(f"checkpoint key collision after remap: {new_key}")
        out[new_key] = value
    return out, changed


def load_compatible_pretrained(
    name: str, *, cache_dir: str | None = None, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    if torch.device(device).type == "cuda":
        # Do not let a broken NVML probe silently turn a learned-model run into
        # a CPU run.  A real allocation is the contract required by the
        # experiment, and fails closed on hosts with driver/device mismatch.
        torch.empty(0, device=device)
    weights_path, config_path = _resolve_local(name, cache_dir=cache_dir)
    config = json.loads(config_path.read_text())
    model = instantiate(config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    remapped, changed = remap_legacy_vit_keys(state_dict)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "incompatible LeWM checkpoint after explicit remap: "
            f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    model._checkpoint_compat = {"weights": str(weights_path), "remapped_keys": changed}
    return model

