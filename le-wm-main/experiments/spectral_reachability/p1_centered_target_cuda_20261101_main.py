"""Encode translated actual history and target; never reuse old root latents."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import e3_cp10_target_cuda_20261031 as encoder
from . import p1_centered_witness_20261101_main as witness
from . import p1_rgb_visibility_20261101_main as visibility

NAME = "p1_centered_target_cuda_20261101_main"
DIRECTORY = witness.old.ROOT / "results/p1_centered_witness_20261101_main_RUN_MAIN01"


def collect():
    report_path = DIRECTORY / "report.json"
    report = json.loads(report_path.read_bytes())
    protocol = json.loads((DIRECTORY / "protocol.json").read_bytes())
    if (report["status"] != "recorded" or not report["task_admitted"]
            or not report["warmup_translation_verified"]
            or protocol["offset"] != list(witness.OFFSET)
            or protocol["phases"] != [list(p) for p in witness.PHASES]
            or protocol["sources"] != report["sources"]
            or [r["step"] for r in report["rows"]] != list(range(21))
            or [r["step"] for r in report["history"]] != [0, 5, 10]):
        raise ValueError("complete translated task")
    checks = witness.old.eligibility([r["state"] for r in report["rows"]])
    if not all(checks.values()) or any(report["checks"].get(k) != v for k, v in checks.items()):
        raise ValueError("physical eligibility")
    sources = dict(report["sources"])
    images = {}
    for row in report["history"] + report["rows"]:
        path = Path(row["png"])
        if path.parent != DIRECTORY or path.is_symlink() or encoder.base.sha(path) != row["file_sha256"]:
            raise ValueError("frame file identity")
        with Image.open(path) as im:
            pixels = np.asarray(im).copy()
        if (hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]
                or visibility.inspect(pixels) != row["visibility"]
                or not row["visibility"]["eligible"]):
            raise ValueError("frame RGB/margin")
        images[str(path)] = pixels
        sources[str(path)] = row["file_sha256"]
    if any(row.get("terminated") or row.get("truncated") for row in report["rows"]):
        raise ValueError("censored task")
    for path in (report_path, DIRECTORY / "protocol.json", Path(__file__).resolve()):
        sources[str(path)] = encoder.base.sha(path)
    if any(encoder.base.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    target = report["rows"][-1]
    pixels = images[target["png"]]
    # No witness actions or intermediate physical states are exposed to the planner.
    entry = dict(root=0, root_state=report["rows"][0]["state"], goal_state=target["state"],
                 pixel_sha256=target["pixel_sha256"],
                 pixels_uint8_base64=base64.b64encode(pixels.tobytes()).decode(),
                 task_eligibility=report["checks"], translated_known_root=True,
                 history_pixel_sha256=[r["pixel_sha256"] for r in report["history"]])
    return entry, [images[r["png"]] for r in report["history"]] + [pixels], sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != encoder.ROOT / "results"
            or output.exists() or not output.name.startswith(NAME)):
        raise ValueError("exclusive output")
    entry, images, sources = collect()
    sources.update(encoder.base.bindings())
    with encoder.base.locking.gpu_lock(encoder.base.LOCK):
        host = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(encoder.base.MODEL), device="cuda:0").eval().requires_grad_(False)
        before = encoder.base.model_hash(model)
        if before != encoder.base.MODEL_HASH:
            raise ValueError("checkpoint hash")
        latents = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for pixels in images:
                image = p0._image_tensor(pixels, transform=p0.image_transform(), device=torch.device("cuda:0"))
                latent = model.encode({"pixels": image})["emb"][0, 0]
                if latent.shape != (192,) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("finite latent")
                latents.append(latent.cpu().tolist())
        torch.cuda.synchronize()
        if encoder.base.model_hash(model) != before or any(encoder.base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model/source drift")
        entry.update(latent=latents[-1], history_latents=latents[:3])
        digest = encoder.base.write(output, dict(schema=NAME, status="complete", rows=[entry],
             sources=sources, host=host, encoder_calls=4, model_before=before, model_after=before,
             P1_admission=False, independent_confirmation=False))
    print(json.dumps(dict(status="complete", sha256=digest, encoder_calls=4)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    run(args.output)
