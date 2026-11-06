"""Compare local DINO implementations after exact HF-to-Meta weight conversion."""
import hashlib
import json
import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - register dataset compression filters
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Dinov2Model

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def convert(source):
    out, used = {}, set()

    def copy(dst, src):
        if src in used:
            raise ValueError(f"duplicated source {src}")
        out[dst] = source[src]
        used.add(src)

    for dst, src in (("cls_token", "embeddings.cls_token"), ("mask_token", "embeddings.mask_token"),
                     ("pos_embed", "embeddings.position_embeddings"),
                     ("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"),
                     ("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"),
                     ("norm.weight", "layernorm.weight"), ("norm.bias", "layernorm.bias")):
        copy(dst, src)
    for i in range(12):
        hf, meta = f"encoder.layer.{i}.", f"blocks.{i}."
        for suffix in ("weight", "bias"):
            keys = [hf + f"attention.attention.{name}.{suffix}" for name in ("query", "key", "value")]
            out[meta + "attn.qkv." + suffix] = torch.cat([source[k] for k in keys], dim=0)
            used.update(keys)
            for dst, src in (("attn.proj.", "attention.output.dense."), ("norm1.", "norm1."),
                             ("norm2.", "norm2."), ("mlp.fc1.", "mlp.fc1."),
                             ("mlp.fc2.", "mlp.fc2.")):
                copy(meta + dst + suffix, hf + src + suffix)
        copy(meta + "ls1.gamma", hf + "layer_scale1.lambda1")
        copy(meta + "ls2.gamma", hf + "layer_scale2.lambda1")
    if used != set(source):
        raise ValueError(f"unmapped HF parameters: {sorted(set(source) - used)}")
    return out


def stats(a, b):
    difference = (a - b).float()
    return {"max_abs": float(difference.abs().max()), "mean_sq": float(difference.square().mean()),
            "relative_l2": float(difference.norm() / b.float().norm().clamp_min(1e-12))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-offset", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / ("results/pc_dinov2_adapter_zero_offset_dev_20261106_codex.json"
                     if args.zero_offset else "results/pc_dinov2_adapter_equivalence_dev_20261106_codex.json")
    if output.exists():
        raise FileExistsError(output)
    official = root / "external/dinov2_official_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    weights = root / ".cache/dinov2/dinov2-small/model.safetensors"
    print("Compare both DINO implementations on identical real images and mapped HF weights.", flush=True)
    src = load_file(str(weights))
    mapped = convert(src)
    meta = torch.hub.load(str(official), "dinov2_vits14", source="local", pretrained=False)
    if args.zero_offset:
        meta.interpolate_offset = 0.0
    meta.load_state_dict(mapped, strict=True)
    meta = meta.cuda().eval().requires_grad_(False)
    hf = Dinov2Model.from_pretrained(str(weights.parent), local_files_only=True).cuda().eval().requires_grad_(False)
    records = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        rows = panel_rows(data, seed=42, count=10)[:3]
        for row in rows:
            image = np.asarray(data["pixels"][int(row)])
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)[None].cuda()
            for size in (224, 518):
                resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
                mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
                normalized = (resized - mean) / std
                with torch.inference_mode():
                    upstream = meta.forward_features(normalized)
                    local = hf(pixel_values=normalized).last_hidden_state
                records.append({"row": int(row), "size": size,
                                "cls": stats(local[:, 0], upstream["x_norm_clstoken"]),
                                "patch": stats(local[:, 1:], upstream["x_norm_patchtokens"])})
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"hf_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                   "meta_interpolation_offset": meta.interpolate_offset,
                   "source_dir": str(official), "mapped_tensor_count": len(src), "records": records,
                   "scope": "same HF weights mapped into local official-code snapshot; not provenance certification"},
                  stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
