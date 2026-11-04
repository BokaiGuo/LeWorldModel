"""Re-encode sealed static goal-factor images independently."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_v4_pusht_rgb_feedback_dev_20261104_codex as run
from experiments import pc_v4_pusht_rgb_goal_factor_diag_v2_20261104_codex as factor

OUT = factor.PREFIX.with_name(factor.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    host, sha = run.HOST, run.HOST.bridge.sha_file
    if os.environ.get("CUDA_VISIBLE_DEVICES") != host.UUID:
        raise ValueError("cooperative GPU UUID required")
    result = json.loads(factor.OUT.read_text())
    run.seal.verify(result["dependencies"])
    if sha(factor.IMAGES) != result["images_sha256"] or len(result["rows"]) != 8:
        raise ValueError("image archive or row count mismatch")
    differences, images_checked = [], 0
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        if host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid) != host.canonical_gpu_uuid(host.UUID):
            raise ValueError("unexpected GPU")
        device = torch.device("cuda:0")
        model = run.saved.load_compatible_pretrained(run.saved.POLICY, device="cuda").eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        model_hash = run.saved.checked.tensor_state_hash(model)
        if model_hash != result["model_hash"]:
            raise ValueError("model mismatch")
        transform = run.saved.p0.image_transform()
        with torch.no_grad(), np.load(factor.IMAGES, allow_pickle=False) as archive:
            if len(archive.files) != 32:
                raise ValueError("unexpected archive size")
            for row in result["rows"]:
                arm, index = row["arm"], row["boundary"]
                with np.load(run.path(arm, f"context{index}.npz"), allow_pickle=False) as context:
                    goal, original = context["goal"].copy(), context["frames"][-1].copy()
                goal_emb = run.saved.p0._goal_latent(model, goal, transform=transform, device=device)
                for name in ("base", "agent", "object", "both"):
                    image = archive[f"{arm}_{index}_{name}"]
                    if run.phys._array_hash(image) != row["rgb_sha256"][name]:
                        raise ValueError("RGB hash mismatch")
                    if name == "base":
                        np.testing.assert_array_equal(image, original)
                    if name == "both":
                        np.testing.assert_array_equal(image, goal)
                    emb = run.saved.p0._goal_latent(model, image, transform=transform, device=device)
                    score = float((emb-goal_emb).square().mean().cpu())
                    differences.append(abs(score-row["goal_mse"][name]))
                    images_checked += 1
        if max(differences) > 2e-6 or run.saved.checked.tensor_state_hash(model) != model_hash:
            raise ValueError("score mismatch or model drift")
    run.seal.verify(result["dependencies"])
    host.write_new(OUT, {"result_sha256": sha(factor.OUT), "images_sha256": sha(factor.IMAGES),
        "audit_script_sha256": sha(Path(__file__)), "model_hash": model_hash,
        "images_checked": images_checked, "max_score_abs_difference": max(differences),
        "cuda_allocation": allocation, "scope": "same encoder, saved images, not independent dynamics"})
    print(f"Re-encoded {images_checked} saved images; maximum score error {max(differences)}.", flush=True)


if __name__ == "__main__":
    main()
