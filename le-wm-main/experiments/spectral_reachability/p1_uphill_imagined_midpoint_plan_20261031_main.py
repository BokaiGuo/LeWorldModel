"""Matched second-packet optimization using model-imagined midpoint."""
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
NAME = "p1_uphill_imagined_midpoint_plan_20261031_main"
PROTOCOL = base.ROOT.parent / "docs/D2P_P1_IMAGINED_MIDPOINT_CONTROL_PROTOCOL_20261031_MAIN.md"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
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
            for row, pair, raw, _actual_midpoint_rgb, frame in entries:
                base.bridge.cp5.blind.validate_row(row)
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_rgb = np.asarray(image).copy()
                with torch.no_grad():
                    cp = model.encode({"pixels": p0._image_tensor(cp_rgb, transform=transform, device=device)})["emb"][0, 0]
                    states, history, goal, mean, std = base.bridge.cp5.context(row, cp, device=device)
                    first = torch.as_tensor(raw[0].copy(), dtype=torch.float32, device=device)
                    ctx = torch.cat((history, ((first-mean)/std).reshape(1, 1, 10)), dim=1)
                    imagined = model.predict(states, model.action_encoder(ctx))[0, -1]
                    states, history, goal, mean, std = prior.feedback.second_context(row, cp, imagined, raw[0])
                action = torch.nn.Parameter(torch.tensor(raw[1][None], dtype=torch.float32, device=device))
                optimizer = torch.optim.Adam([action], lr=0.05)

                def score():
                    token = ((action-mean)/std).reshape(1, 1, 10)
                    prediction = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                    return (prediction-goal).square().mean()

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
                    with torch.no_grad():
                        action.clamp_(-1, 1)
                with torch.no_grad():
                    final = float(score())
                if not np.isfinite([baseline, final]).all():
                    raise ValueError("score")
                torch.cuda.synchronize(device)
                chosen = action.detach().cpu().numpy()[0].astype("<f4") if final < baseline else raw[1].copy()
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
                  "model_before": before, "model_after": after,
                  "per_row_calls_code_path": {"encoder": 1, "predict": 23, "action_encoder": 23, "backward": 20},
                  "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
                  "limitations": "Known roots, post-outcome development, model-imagined midpoint."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replan all fixed second packets using imagined midpoints on host CUDA.", flush=True)
    run(args.output)
