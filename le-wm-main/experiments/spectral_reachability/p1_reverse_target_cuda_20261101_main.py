"""Audit the fixed V3 task witness and encode only its final RGB on CUDA."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import e3_cp10_target_cuda_20261031 as encoder
from . import p1_reverse_witness_20261101_main as witness

NAME = "p1_reverse_target_cuda_20261101_main"
ROOT = encoder.ROOT
DIRECTORY = ROOT / "results/p1_reverse_witness_v3_20261101_main_RUN_MAIN01"
REPORT_SHA = "4ca900a77b1e69e840a317fda14da11da505c41000d5daeef30a9b247f98d27b"
DOC = ROOT.parent / "docs/D2P_P1_REVERSE_TARGET_PROTOCOL_20261101_MAIN.md"


def collect():
    report_path = DIRECTORY / "report.json"
    need = encoder.qualification.require
    need(encoder.base.sha(report_path) == REPORT_SHA, "fixed eligible witness")
    report = json.loads(report_path.read_bytes())
    binding_path = DIRECTORY / "driver_binding.json"
    binding = json.loads(binding_path.read_bytes())
    need(binding["report_sha256"] == REPORT_SHA
         and encoder.base.sha(binding["driver"]) == binding["driver_sha256"]
         and encoder.base.sha(DIRECTORY / "protocol.json") == binding["protocol_sha256"],
         "driver binding")
    protocol = json.loads((DIRECTORY / "protocol.json").read_bytes())
    need(protocol["phases"] == binding["phases"], "phase binding")
    rows = report["rows"]
    need(report["status"] == "witness_recorded" and report["task_eligible"]
         and [r["step"] for r in rows] == list(range(21))
         and not any(r["terminated"] or r["truncated"] for r in rows), "complete witness")
    states = np.asarray([r["state"] for r in rows], np.float64)
    commands = np.asarray([r["command"] for r in rows[1:]], np.float32)
    checks = witness.eligibility(states)
    need(checks == report["checks"] and all(checks.values())
         and np.isfinite(states).all() and np.isfinite(commands).all()
         and commands.shape == (20, 2) and np.max(np.abs(commands)) <= 1
         and ((states[:, :4] >= 0) & (states[:, :4] <= 512)).all(), "independent eligibility")
    sources = dict(report["sources"])
    sources[binding["driver"]] = binding["driver_sha256"]
    pixels = None
    for row in rows:
        path = Path(row["png"])
        need(path.parent == DIRECTORY and not path.is_symlink()
             and encoder.base.sha(path) == row["file_sha256"], "frame binding")
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
             and hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "frame pixels")
        sources[str(path)] = row["file_sha256"]
    target = report["target"]
    need(target["state"] == rows[-1]["state"]
         and target["pixel_sha256"] == rows[-1]["pixel_sha256"]
         and base64.b64decode(target["pixels_uint8_base64"], validate=True) == pixels.tobytes(),
         "final target identity")
    for path in (report_path, binding_path, DIRECTORY / "protocol.json",
                 DOC, Path(__file__).resolve(), Path(witness.__file__).resolve()):
        sources[str(path)] = encoder.base.sha(path)
    need(all(encoder.base.sha(p) == h for p, h in sources.items()), "sources")
    # No witness commands or intermediate states are returned to the planner adapter.
    return [dict(root=0, root_state=rows[0]["state"], goal_state=target["state"],
                 pixels=pixels, pixel_sha256=target["pixel_sha256"],
                 pixels_uint8_base64=target["pixels_uint8_base64"],
                 task_eligibility=checks, privileged_task_generation=True)], sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if not out.is_absolute() or out.parent != ROOT / "results" or out.exists():
        raise ValueError("exclusive output")
    entries, sources = collect()
    sources.update(encoder.base.bindings())
    with encoder.base.locking.gpu_lock(encoder.base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(encoder.base.MODEL), device=device).eval().requires_grad_(False)
        before = encoder.base.model_hash(model)
        if before != encoder.base.MODEL_HASH:
            raise ValueError("checkpoint")
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for entry in entries:
                image = p0._image_tensor(entry["pixels"], transform=p0.image_transform(), device=device)
                latent = model.encode({"pixels": image})["emb"][0, 0]
                if latent.shape != (192,) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("finite goal embedding")
                rows.append({**{k: v for k, v in entry.items() if k != "pixels"},
                             "latent": latent.cpu().tolist()})
        torch.cuda.synchronize()
        if encoder.base.model_hash(model) != before or any(encoder.base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source/model drift")
        digest = encoder.base.write(out, dict(schema=NAME, status="complete", host=host,
            rows=rows, sources=sources, model_before=before, model_after=before,
            encoder_calls=len(rows), P1_admission=False, independent_confirmation=False,
            physical_execution=False))
    print(json.dumps(dict(status="complete", sha256=digest, rows=len(rows))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    run(args.output)
