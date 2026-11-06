"""Evaluation-only latent predictions; read terminal RGB only after prediction."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.pc_jepa_v7_isolated_worker_20261104_codex import ROOT, LOCK, UUID, CHUNK, sha
from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history


def main():
    base = Path(sys.argv[1])
    inp = base / "input"
    metadata = json.loads((inp / "model_metadata.json").read_text())
    with np.load(inp / "context.npz", allow_pickle=False) as data:
        context = {k: data[k].copy() for k in data.files}
    actions = np.load(inp / "bank.npy", allow_pickle=False)
    if (actions.shape != (64, 15, 2) or actions.dtype != np.float32
            or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1):
        raise ValueError("invalid fixed bank")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("CUDA device 0 required")
    if str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") != UUID:
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
            patches = encode(context["rgb_history"])
            goal = encode(context["goal_rgb"])
            visual = patches.reshape(1, 3, 1, 16, 16, 384)
            prop = torch.from_numpy(context["normalized_proprio"]).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            history = torch.from_numpy(context["normalized_history_packets"]).unsqueeze(0).to("cuda")
            mean, std = (torch.as_tensor(context[k], device="cuda") for k in ("action_mean", "action_std"))
            if torch.any(std <= 0):
                raise ValueError("invalid action standard deviation")
            core = NativeCore(models["predictor"])
            predictions = []
            for offset in range(0, len(actions), CHUNK):
                chunk = torch.as_tensor(actions[offset:offset + CHUNK], device="cuda")
                packet = ((chunk - mean) / std).reshape(len(chunk), -1, 10).transpose(0, 1)
                pred = unroll_history(core, visual, features, history, packet, 3)["visual"][-1]
                predictions.append(pred.reshape(len(chunk), 256, 384).cpu().numpy())
            predicted = np.concatenate(predictions)
            # The evaluation labels are not read until every predicted latent is fixed.
            with np.load(inp / "images.npz", allow_pickle=False) as data:
                if set(data.files) != {"goal_rgb", "terminal_rgb"}:
                    raise ValueError("unexpected image fields")
                labels = data["terminal_rgb"].copy()
                np.testing.assert_array_equal(data["goal_rgb"], context["goal_rgb"][0])
            if labels.shape != (64, 224, 224, 3) or labels.dtype != np.uint8:
                raise ValueError("invalid terminal images")
            actual = np.concatenate([encode(labels[i:i+8]).cpu().numpy() for i in range(0, 64, 8)])
            torch.cuda.synchronize()
        if before != {key: v4.model_receipt(model) for key, model in models.items()}:
            raise ValueError("model changed")
    with (base / "output/latents.npz").open("xb") as handle:
        np.savez_compressed(handle, predicted=predicted, actual=actual,
                            current=patches[-1].cpu().numpy(), goal=goal[0].cpu().numpy())
    output = {"status": "completed", "worker_sha256": sha(Path(__file__)),
              "context_sha256": sha(inp / "context.npz"), "bank_sha256": sha(inp / "bank.npy"),
              "images_sha256": sha(inp / "images.npz"), "metadata_sha256": sha(inp / "model_metadata.json"),
              "latents_sha256": sha(base / "output/latents.npz"), "predictor_module_calls": core.calls,
              "prediction_token_samples": 192, "strict_load": strict, "imported_sources": sources}
    with (base / "output/receipt.json").open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Computed 64 latent predictions and encoded terminal labels after prediction.", flush=True)


if __name__ == "__main__":
    main()
