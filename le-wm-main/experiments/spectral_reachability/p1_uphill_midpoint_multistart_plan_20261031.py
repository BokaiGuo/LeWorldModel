"""Replan every fixed midpoint with four bounded Adam starts on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_midpoint_feedback_plan_20261031_main as prior

base = prior.base
NAME = "p1_uphill_midpoint_multistart_plan_20261031"
PROTOCOL = base.ROOT.parent / "docs/D2P_P1_MIDPOINT_MULTISTART_PROTOCOL_20261031_MAIN.md"


def initial_actions(pair, original):
    if type(pair) is not int or not 0 <= pair < 8:
        raise ValueError("logical pair")
    raw = np.asarray(original, dtype="<f4")
    if raw.shape != (5, 2) or not np.isfinite(raw).all() or (np.abs(raw) > 1).any():
        raise ValueError("original action")
    random = np.random.default_rng(20261031 + pair).uniform(-1, 1, size=(2, 5, 2)).astype("<f4")
    return np.stack((raw, np.zeros_like(raw), *random))


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
                    def encode(image):
                        return model.encode({"pixels": p0._image_tensor(
                            image, transform=transform, device=device)})["emb"][0, 0]
                    states, history, goal, mean, std = prior.feedback.second_context(
                        row, encode(cp_rgb), encode(midpoint_rgb), raw[0])
                def score(action):
                    token = ((action - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                    return (pred - goal).square().mean()
                torch.cuda.synchronize(device)
                started = time.monotonic()
                starts = initial_actions(row["logical_pair"], raw[1])
                with torch.no_grad():
                    baseline = float(score(torch.as_tensor(raw[1][None], device=device)))
                candidates, scores = [], []
                for init in starts:
                    action = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([action], lr=0.05)
                    for _ in range(20):
                        optimizer.zero_grad(set_to_none=True)
                        score(action).backward()
                        if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                            raise ValueError("gradient")
                        optimizer.step()
                        with torch.no_grad():
                            action.clamp_(-1, 1)
                    with torch.no_grad():
                        scores.append(float(score(action)))
                    candidates.append(action.detach()[0].cpu().numpy().astype("<f4").copy())
                if not np.isfinite([baseline, *scores]).all():
                    raise ValueError("nonfinite score")
                index = min(range(4), key=lambda i: (scores[i], i))
                optimized = scores[index]
                chosen = candidates[index] if optimized < baseline else raw[1].copy()
                torch.cuda.synchronize(device)
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "first_packet_hex": raw[0].tobytes().hex(), "second_packet_hex": chosen.tobytes().hex(),
                             "midpoint_pixel_sha256": frame["pixel_sha256"], "baseline_score": baseline,
                             "optimized_score": optimized, "chosen": "optimized" if optimized < baseline else "original",
                             "candidate_scores": scores, "candidate_hex": [a.tobytes().hex() for a in candidates],
                             "selected_start": index, "start_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
                             "optimizer_seconds": time.monotonic()-started})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        digest = base.write(output, {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                                     "host": host, "model_before": before, "model_after": after,
                                     "per_row_calls_code_path": {"encoder": 2, "predict": 85,
                                                                  "action_encoder": 85, "backward": 80},
                                     "P1_admission": False, "independent_confirmation": False,
                                     "equal_budget_certified": False,
                                     "limitations": "Known roots, post-outcome design, extra optimization."})
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Replan all fixed real midpoints with four Adam starts on host CUDA.", flush=True)
    run(parser.parse_args().output)
