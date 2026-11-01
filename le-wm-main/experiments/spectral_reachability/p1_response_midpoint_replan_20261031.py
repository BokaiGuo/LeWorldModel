"""Replan the second packet of every verified complete response trajectory."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_response_cp5_same_action_20261031 as source

ROOT, need, sha = source.ROOT, source.need, source.sha
NAME = "p1_response_midpoint_replan_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_MIDPOINT_REPLAN_PROTOCOL_20261031.md"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    entries, excluded, sources = source.collect()
    base = source.common.base
    for path, digest in base.bindings().items():
        need(path not in sources or sources[path] == digest, "source collision")
        sources[path] = digest
    for path in (Path(__file__).resolve(), PROTOCOL):
        sources[str(path)] = sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen checkpoint")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                blind, raw = entry["blind"], entry["raw"]
                with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as img:
                    cp_pixels = np.asarray(img.convert("RGB")).copy()
                with torch.no_grad():
                    def encode(pixels):
                        return model.encode({"pixels": p0._image_tensor(
                            pixels, transform=transform, device=device)})["emb"][0, 0]
                    cp, midpoint = encode(cp_pixels), encode(entry["mid"])
                    states, history, goal, mean, std = source.feedback.second_context(
                        blind, cp, midpoint, raw[0])
                action = torch.nn.Parameter(torch.tensor(raw[1][None], dtype=torch.float32, device=device))
                optimizer = torch.optim.Adam([action], lr=0.05)

                def score():
                    token = ((action - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                    return (pred - goal).square().mean()

                torch.cuda.synchronize(device)
                started = time.monotonic()
                with torch.no_grad():
                    baseline = float(score())
                for _ in range(20):
                    optimizer.zero_grad(set_to_none=True)
                    loss = score()
                    loss.backward()
                    need(action.grad is not None and bool(torch.isfinite(action.grad).all()), "gradient")
                    optimizer.step()
                    with torch.no_grad():
                        action.clamp_(-1, 1)
                with torch.no_grad():
                    final = float(score())
                need(np.isfinite([baseline, final]).all(), "finite score")
                chosen = action.detach().cpu().numpy()[0].astype("<f4") if final < baseline else raw[1].copy()
                need(chosen.shape == (5, 2) and np.isfinite(chosen).all() and
                     (np.abs(chosen) <= 1).all(), "bounded action")
                torch.cuda.synchronize(device)
                rows.append({
                    **{k: entry[k] for k in ("index", "logical_pair", "condition", "arm", "role", "root_slot")},
                    "first_packet_hex": raw[0].tobytes().hex(),
                    "original_second_packet_hex": raw[1].tobytes().hex(),
                    "second_packet_hex": chosen.tobytes().hex(),
                    "midpoint_pixel_sha256": hashlib.sha256(entry["mid"].tobytes()).hexdigest(),
                    "original_action_sha256": entry["action_sha256"],
                    "baseline_score": baseline, "optimized_score": final,
                    "chosen": "optimized" if final < baseline else "original",
                    "optimizer_seconds": time.monotonic() - started,
                })
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(before == after and all(sha(p) == d for p, d in sources.items()), "model/source drift")
        result = {
            "schema": NAME, "status": "complete", "rows": rows, "excluded": excluded,
            "fixed_rows": 16, "sources": sources, "host": host, "model_before": before, "model_after": after,
            "per_row_calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20},
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots, post-outcome exploration, offline midpoint, extra optimization; "
                           "no physical execution of replanned actions.",
        }
        digest = base.write(out, result)
    print(json.dumps({"rows": len(rows), "excluded": len(excluded), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replan verified actual trajectory midpoints on host CUDA; physical validation remains required.",
          flush=True)
    run(args.output)
