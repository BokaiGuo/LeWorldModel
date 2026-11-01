"""Compare fixed planning goal latents with the corresponding real goal RGB encoding."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_rgb_latent_task_compare_20261030_v1 as source

NAME = "codex_p1_goal_rgb_binding_20261030_main"


def validate_identity(index, row, item):
    identity = item["identity"]
    if (row["index"] != item["index"] or row["index"] != index
            or row["logical_pair"] != identity["logical_pair"] or row["role"] != identity["role"]):
        raise ValueError("identity mismatch")
    return identity


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    saved, images = source.inputs()
    bundle, _ = gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if len(bundle["rows"]) != 24 or len(images) != 24:
        raise ValueError("fixed roster")
    bindings = base.bindings()
    for path in (Path(__file__).resolve(), Path(source.__file__).resolve(),
                 Path(source.source.__file__).resolve(), source.RGB_SAVED, base.BUNDLE):
        bindings[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model mismatch")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for index, (row, item, pair) in enumerate(zip(bundle["rows"], saved["rows"], images, strict=True)):
                identity = validate_identity(index, row, item)
                goal_rgb = pair[1]
                goal = model.encode({"pixels": p0._image_tensor(goal_rgb, transform=transform, device=device)})["emb"]
                if goal.shape != (1, 1, 192):
                    raise ValueError("goal shape")
                stored = torch.as_tensor(row["goal_latent"], dtype=torch.float32, device=device)
                diff = goal[0, 0] - stored
                if not bool(torch.isfinite(diff).all()):
                    raise ValueError("nonfinite difference")
                rows.append({"index": index, "identity": identity,
                             "goal_pixel_sha256": hashlib.sha256(goal_rgb.tobytes()).hexdigest(),
                             "max_abs_difference": float(diff.abs().max()),
                             "mse": float(diff.square().mean()),
                             "stored_goal_norm": float(stored.norm()),
                             "encoded_goal_norm": float(goal[0, 0].norm())})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(Path(p)) != h for p, h in bindings.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": bindings,
                  "host": host, "model_before": before, "model_after": before,
                  "encode_calls": 24, "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Known roots; same encoder path; source checks are not a full dependency closure."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest,
                      "max_difference": max(r["max_abs_difference"] for r in rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Check saved goal latents against task RGB encodings on host CUDA.", flush=True)
    run(parser.parse_args().output)
