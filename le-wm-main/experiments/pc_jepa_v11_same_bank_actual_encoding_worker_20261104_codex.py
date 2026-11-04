"""Encode sanitized terminal RGBs for a saved-bank diagnostic in isolation."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.pc_jepa_v7_isolated_worker_20261104_codex import ROOT, LOCK, UUID, sha
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock


def main():
    base = Path(sys.argv[1])
    inp = base / "input"
    metadata = json.loads((inp / "model_metadata.json").read_text())
    with np.load(inp / "images.npz", allow_pickle=False) as package:
        if set(package.files) != {"goal_rgb", "terminal_rgb"}:
            raise ValueError("unexpected encoder input")
        goal, frames = package["goal_rgb"].copy(), package["terminal_rgb"].copy()
    if (goal.shape != (224, 224, 3) or frames.shape != (64, 224, 224, 3)
            or goal.dtype != np.uint8 or frames.dtype != np.uint8):
        raise ValueError("unexpected image shape or dtype")
    if not torch.cuda.is_available() or str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") != UUID:
        raise ValueError("unexpected GPU")
    with gpu_lock(LOCK):
        models, strict, sources, before = native.load_cuda(ROOT, metadata)
        _, v5, _ = native.helper(ROOT).dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import SDPBackend, sdpa_kernel
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            def encode(images):
                pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                return models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            reference = encode(goal[None]).reshape(1, -1)
            scores = []
            for offset in range(0, len(frames), 8):
                features = encode(frames[offset:offset + 8]).reshape(-1, reference.shape[1])
                scores.extend((features - reference).square().mean(dim=1).cpu().numpy().tolist())
            torch.cuda.synchronize()
        if before != {key: v4.model_receipt(model) for key, model in models.items()}:
            raise ValueError("model changed")
    output = {"image_sha256": sha(inp / "images.npz"), "metadata_sha256": sha(inp / "model_metadata.json"),
              "worker_sha256": sha(Path(__file__)), "scores": scores,
              "strict_load": strict, "imported_sources": sources}
    with (base / "output/encoded.json").open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"encoded {len(scores)} physical terminal images", flush=True)


if __name__ == "__main__":
    main()
