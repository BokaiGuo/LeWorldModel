"""Guarded, fixed-roster best-iterate trust Adam development experiment."""
import argparse
import base64
import io
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_feedback_plan_20261030_main as source
from . import codex_p1_midpoint_stop_guard_20261030_v1 as guard

base = source.base
NAME = "codex_p1_adam_best_iterate_20261030_main"
LAMBDA = 0.1


def select_best(scores):
    if len(scores) != 21 or not all(math.isfinite(x) for x in scores):
        raise ValueError("finite 21-step objective history required")
    return min(range(len(scores)), key=lambda i: (scores[i], i))


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    entries = guard.verified_saved_inputs()
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint mismatch")
        tx = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, physical, raw, rgb, frame in entries:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                with torch.no_grad():
                    def encode(x):
                        return model.encode({"pixels": p0._image_tensor(x, transform=tx, device=device)})["emb"][0, 0]
                    cp5, midpoint = encode(cp5_rgb), encode(rgb)
                    states, history, goal, mean, std = source.feedback.second_context(row, cp5, midpoint, raw[0])
                reference = torch.tensor(raw[1][None], dtype=torch.float32, device=device)
                action = torch.nn.Parameter(reference.clone())
                optimizer = torch.optim.Adam([action], lr=0.05)

                def score():
                    token = ((action - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                    return (pred-goal).square().mean() + LAMBDA*(action-reference).square().mean()

                torch.cuda.synchronize(device)
                started = time.monotonic()
                with torch.no_grad():
                    baseline = float(score())
                scores, packets = [], []
                for _ in range(20):
                    optimizer.zero_grad(set_to_none=True)
                    objective = score()
                    scores.append(float(objective.detach()))
                    packets.append(action.detach().clone())
                    objective.backward()
                    if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                        raise ValueError("invalid gradient")
                    optimizer.step()
                    with torch.no_grad():
                        action.clamp_(-1, 1)
                with torch.no_grad():
                    scores.append(float(score()))
                    packets.append(action.detach().clone())
                best = select_best(scores)
                if abs(scores[0] - baseline) > 1e-6:
                    raise ValueError("baseline mismatch")
                packet = packets[best].cpu().numpy()[0].astype("<f4")
                torch.cuda.synchronize(device)
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "first_packet_hex": raw[0].tobytes().hex(), "second_packet_hex": packet.tobytes().hex(),
                             "midpoint_pixel_sha256": frame["pixel_sha256"], "baseline_score": baseline,
                             "optimized_score": scores[-1], "objective_history": scores, "selected_step": best,
                             "selected_score": scores[best], "chosen": "original" if best == 0 else "optimized",
                             "optimizer_seconds": time.monotonic() - started})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model):
            raise ValueError("model mutation")
    result = {"schema": NAME, "status": "complete", "rows": rows, "host": host,
              "model_before": before, "model_after": before, "code_sha256": base.sha(Path(__file__).resolve()),
              "guard_sha256": base.sha(Path(guard.__file__).resolve()),
              "source_sha256": base.sha(Path(source.__file__).resolve()), "penalty_weight": LAMBDA,
              "per_row_calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20},
              "P1_admission": False, "independent_confirmation": False,
              "limitations": "known roots; post-outcome development; offline midpoint; path counts not full budget"}
    print(json.dumps({"rows": len(rows), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    out = Path(ap.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Guard and optimize 24 fixed real-RGB midpoint suffixes on host CUDA.", flush=True)
    run(out)
