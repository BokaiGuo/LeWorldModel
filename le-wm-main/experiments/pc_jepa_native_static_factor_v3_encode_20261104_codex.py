"""Encode sealed post-hoc image factors using the frozen JEPA visual encoder."""
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from experiments.pc_jepa_native_executed_packet_diag_20261104_codex import ROOT, SOURCE, LOCK, UUID, sha
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock

OUT = ROOT / "results/pc_jepa_native_static_factor_v3_20261104_codex"


def main():
    destination, arrays_path = OUT / "scores.json", OUT / "features.npz"
    if destination.exists() or arrays_path.exists():
        raise FileExistsError("existing score or feature output")
    render_path, image_path = OUT / "render.json", OUT / "images.npz"
    render = json.loads(render_path.read_text())
    if render["status"] != "render_complete" or render["images_sha256"] != sha(image_path):
        raise ValueError("unsealed image set")
    for path, expected in render["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"source changed: {path}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("GPU 0 required")
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    if uuid != UUID:
        raise ValueError("unexpected GPU")
    record = {"status": "running", "scope": "post-hoc static sensitivity; no planner or physical execution",
              "render_sha256": sha(render_path), "images_sha256": sha(image_path),
              "script_sha256": sha(Path(__file__)), "gpu_uuid": uuid, "rows": []}
    error = None
    try:
        with gpu_lock(LOCK):
            metadata_path = SOURCE / "response_0/input/model_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            models, strict, sources, before = native.load_cuda(ROOT, metadata)
            _, v5, _ = native.helper(ROOT).dependencies(ROOT)
            v4 = v5.import_v4(ROOT)
            from app.plan_common.datasets.transforms import make_transforms
            from torch.nn.attention import SDPBackend, sdpa_kernel
            transform = make_transforms(img_size=224, random_horizontal_flip=False,
                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                reprob=0., auto_augment=False, motion_shift=False)
            features = {}
            with np.load(image_path, allow_pickle=False) as images:
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    for key in sorted(images.files):
                        pixels = torch.from_numpy(images[key].copy()).permute(2, 0, 1).unsqueeze(0).to(
                            "cuda", torch.float32) / 255.
                        patch = models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
                        features[key] = patch.squeeze(0).cpu().numpy()
            torch.cuda.synchronize()
            for arm in ("response", "cem"):
                for frame in (10, 15, 20, 25):
                    for mode in ("native", "no_contacts"):
                        goal = features[f"goal_{mode}"].astype(np.float64)
                        scores = {name: float(np.mean((features[f"{arm}_{frame}_{name}_{mode}"].astype(
                            np.float64) - goal) ** 2)) for name in ("base", "agent", "object", "both")}
                        record["rows"].append({"arm": arm, "frame": frame, "mode": mode, "mse": scores,
                            "agent_effect": scores["agent"] - scores["base"],
                            "object_effect": scores["object"] - scores["base"],
                            "interaction": scores["both"] - scores["agent"] - scores["object"] + scores["base"]})
            record.update(strict_load=strict, imported_sources=sources,
                          metadata_sha256=sha(metadata_path),
                          model_unchanged=before == {k: v4.model_receipt(m) for k, m in models.items()})
            if not record["model_unchanged"]:
                raise ValueError("model changed")
            for path, expected in render["bindings"].items():
                if sha(Path(path)) != expected:
                    raise ValueError(f"source changed during scoring: {path}")
            with arrays_path.open("xb") as f:
                np.savez_compressed(f, **features)
            record.update(status="complete", features_sha256=sha(arrays_path))
    except Exception:
        error = traceback.format_exc()
        record.update(status="failed", error=error)
    with destination.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
