"""Re-encode two previously observed E3 cp5 targets on host CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import e3_initial_goal_qualification_20261031 as qualification

ROOT = base.ROOT
NAME = "e3_cp5_target_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_CP5_TARGET_CUDA_PROTOCOL_20261031.md"
SOURCE = ROOT / "results/agent_p3_h3_witness_physical_20261029_v1_MAIN01"


def collect():
    if base.sha(qualification.INPUT) != qualification.INPUT_SHA:
        raise ValueError("resolver source")
    data = json.loads(qualification.INPUT.read_bytes())
    qualified = qualification.analyze(data)
    entries = []
    sources = {str(qualification.INPUT): qualification.INPUT_SHA,
               str(qualification.ENV_SOURCE): qualification.ENV_SHA,
               str(PROTOCOL): base.sha(PROTOCOL), str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())}
    for i, item in enumerate(data["root_provenance"]):
        if qualified["exploratory_cp5_candidates"][i]["initial_success"]:
            raise ValueError("target initially successful")
        record = item["provenance"]["root_native_branches"][0]["c5"]
        path = SOURCE / record["png"]
        if path.name != record["png"] or base.sha(path) != record["file_sha256"]:
            raise ValueError("source frame")
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8 or \
                hashlib.sha256(pixels.tobytes()).hexdigest() != record["pixel_sha256"]:
            raise ValueError("pixel identity")
        existing = np.asarray(item["provenance"]["actual_cp5_task"]["root_context"]["latent_history"][-1],
                              dtype=np.float32)
        if existing.shape != (192,) or not np.isfinite(existing).all():
            raise ValueError("saved latent")
        sources[str(path)] = record["file_sha256"]
        entries.append({"root": i, "pixels": pixels, "existing": existing,
                        "pixel_sha256": record["pixel_sha256"],
                        "witness": item["provenance"]["prefix5"]})
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("source drift")
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_") \
            or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    entries, sources = collect()
    sources.update(base.bindings())
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for item in entries:
                latent = model.encode({"pixels": p0._image_tensor(item["pixels"], transform=transform,
                                                                   device=device)})["emb"][0, 0]
                actual = latent.detach().cpu().numpy().astype(np.float32)
                difference = actual.astype(np.float64) - item["existing"].astype(np.float64)
                rows.append({"root": item["root"], "pixel_sha256": item["pixel_sha256"],
                             "latent": actual.tolist(), "existing": item["existing"].tolist(),
                             "max_abs_difference": float(np.max(np.abs(difference))),
                             "mse": float(np.mean(difference ** 2)),
                             "exact_float32": actual.tobytes() == item["existing"].tobytes(),
                             "prefix5": item["witness"]})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model) or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("model or source drift")
        digest = base.write(out, {"schema": NAME, "status": "complete", "host": host,
                                  "model_before": before, "model_after": before, "rows": rows,
                                  "sources": sources, "encoder_calls": 2,
                                  "scientific_admission": False, "independent_confirmation": False,
                                  "limitations": "Known observed cp5 goals; no planner or physical execution."})
    print(json.dumps({"sha256": digest, "max_abs": [x["max_abs_difference"] for x in rows]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Re-encode the two witnessed cp5 goal images with the frozen model on CUDA.", flush=True)
    run(parser.parse_args().output)
