"""Encode every available fixed-candidate real step; no post-stop imputation."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_all_candidates_midpoint_20261031 as prior

NAME = "p1_candidate_step_rgb_cuda_20261031"
ROOT, base = prior.ROOT, prior.base
PROTOCOL = ROOT.parent / "docs/D2P_P1_CANDIDATE_REAL_TIME_RGB_PROTOCOL_20261031.md"


def collect():
    entries, sources = prior.inputs()
    sources.update(base.bindings())
    sources[str(PROTOCOL)] = base.sha(PROTOCOL)
    sources[str(Path(__file__).resolve())] = base.sha(Path(__file__).resolve())
    for entry in entries:
        kind, index = entry["kind"], entry["index"]
        name, _, _ = prior.REPORTS[kind]
        suffix = "" if kind == 0 else f"_r{kind-1}"
        directory = ROOT / f"results/{name}_run{suffix}_RUN01" / f"row_{index:02d}"
        physical = json.loads((directory / "report.json").read_bytes())["row"]
        responses = physical["responses"]
        if len(responses) != entry["returned_suffix"] or [r["step"] for r in responses] != list(
                range(1, len(responses) + 1)):
            raise ValueError("physical response roster")
        frames = []
        for step, response in enumerate(responses, 1):
            matches = [f for f in physical["trajectory"] if f["phase"] == "probe" and f["step"] == step
                       and f["condition"] == physical["condition"]
                       and f["overlay_index"] == (1 if physical["condition"] == "on" else 0)]
            if len(matches) != 1 or matches[0]["state"] != response["state"]:
                raise ValueError("same-time physical frame")
            frame = matches[0]
            path = directory / frame["png"]
            if path.name != frame["png"] or base.sha(path) != frame["file_sha256"]:
                raise ValueError("frame file")
            with Image.open(path) as image:
                pixels = np.asarray(image).copy()
            if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8 or \
                    hashlib.sha256(pixels.tobytes()).hexdigest() != frame["pixel_sha256"]:
                raise ValueError("frame pixel")
            sources[str(path)] = frame["file_sha256"]
            frames.append({"pixels": pixels, "pixel_sha256": frame["pixel_sha256"],
                           "state": response["state"], "terminated": response["terminated"],
                           "truncated": response["truncated"]})
        entry["frames"] = frames
    if len(entries) != 96 or any(base.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("fixed source drift")
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" or \
            not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink():
        raise ValueError("exclusive output")
    entries, sources = collect()
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
            for entry in entries:
                row = entry["blind"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_pixels = np.asarray(image).copy()
                cp = model.encode({"pixels": p0._image_tensor(cp_pixels, transform=transform,
                                                               device=device)})["emb"][0, 0]
                _, _, goal, _, _ = base.bridge.cp5.context(row, cp, device=device)
                steps = []
                for step, frame in enumerate(entry["frames"], 1):
                    z = model.encode({"pixels": p0._image_tensor(frame["pixels"], transform=transform,
                                                                 device=device)})["emb"][0, 0]
                    score = float((z-goal[0]).square().mean())
                    if not np.isfinite(score):
                        raise ValueError("nonfinite goal score")
                    steps.append({"step": step, "actual_goal_mse": score,
                                  "pixel_sha256": frame["pixel_sha256"], "state": frame["state"],
                                  "terminated": frame["terminated"], "truncated": frame["truncated"]})
                rows.append({"index": entry["index"], "kind": entry["kind"],
                             "candidate": entry["candidate"], "role": row["role"],
                             "logical_pair": row["logical_pair"], "model_terminal_score": entry["score"],
                             "action_sha256": hashlib.sha256(entry["action"].tobytes()).hexdigest(),
                             "status": entry["status"], "success": entry["success"], "steps": steps})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": after,
                  "encoder_calls": len(rows) + sum(len(r["steps"]) for r in rows),
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Observed future RGB is offline-only, known roots and repeated actions; "
                                 "no post-stop imputation, no selector or causal performance claim."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "frames": result["encoder_calls"] - len(rows),
                      "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Encode every fixed candidate's observed RGB steps on host CUDA, preserving stops.", flush=True)
    run(parser.parse_args().output)
