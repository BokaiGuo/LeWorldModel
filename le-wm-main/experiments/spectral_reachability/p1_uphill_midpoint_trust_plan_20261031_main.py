"""Optimize all fixed second packets around their original action in a trust region."""
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_midpoint_feedback_plan_20261031_main as prior

base = prior.base
NAME = "p1_uphill_midpoint_trust_plan_20261031_main"
PROTOCOL = base.ROOT.parent / "docs/D2P_P1_MIDPOINT_TRUST_REGION_PROTOCOL_20261031_MAIN.md"
RADIUS = 0.5


def project(action, origin):
    import torch
    with torch.no_grad():
        action.clamp_(-1, 1)
        delta = action - origin
        norm = torch.linalg.vector_norm(delta)
        action.copy_(origin + delta * torch.clamp(RADIUS / norm.clamp_min(1e-12), max=1))


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if (not out.is_absolute() or out.parent != base.ROOT / "results"
            or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("output")
    entries, sources = prior.inputs()
    sources.update(base.bindings())
    for path in (Path(__file__).resolve(), Path(prior.__file__).resolve(),
                 Path(prior.feedback.__file__).resolve(), PROTOCOL):
        sources[str(path)] = base.sha(path)
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
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, pair, raw, midpoint_rgb, frame in entries:
                base.bridge.cp5.blind.validate_row(row)
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_rgb = np.asarray(image).copy()
                with torch.no_grad():
                    def encode(im):
                        return model.encode({"pixels": p0._image_tensor(
                            im, transform=transform, device=device)})["emb"][0, 0]
                    states, history, goal, mean, std = prior.feedback.second_context(
                        row, encode(cp_rgb), encode(midpoint_rgb), raw[0])
                origin = torch.tensor(raw[1][None], dtype=torch.float32, device=device)
                action = torch.nn.Parameter(origin.clone())
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
                    if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                        raise ValueError("gradient")
                    optimizer.step()
                    project(action, origin)
                with torch.no_grad():
                    final = float(score())
                if not np.isfinite([baseline, final]).all():
                    raise ValueError("score")
                chosen = action.detach().cpu().numpy()[0].astype("<f4") if final < baseline else raw[1].copy()
                if np.linalg.norm(chosen.astype(np.float64) - raw[1]) > RADIUS + 1e-6:
                    raise ValueError("trust radius")
                torch.cuda.synchronize(device)
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "first_packet_hex": raw[0].tobytes().hex(), "second_packet_hex": chosen.tobytes().hex(),
                             "midpoint_pixel_sha256": frame["pixel_sha256"], "baseline_score": baseline,
                             "optimized_score": final, "chosen": "optimized" if final < baseline else "original",
                             "optimizer_seconds": time.monotonic()-started})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": after, "radius": RADIUS,
                  "per_row_calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20},
                  "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
                  "limitations": "Post-outcome design, known roots, offline midpoint, exploratory trust radius."}
        digest = base.write(out, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Trust-region replan fixed second packets from real midpoints on host CUDA.", flush=True)
    run(args.output)
