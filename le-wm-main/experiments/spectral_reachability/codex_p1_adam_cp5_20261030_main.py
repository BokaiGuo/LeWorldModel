"""Fixed four-start Adam suffix exploration on known-root host CUDA inputs."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_multistart_cp5_20261030_main as prior

NAME = "codex_p1_adam_cp5_20261030_main"
STEPS = 20
LR = 0.05


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = prior.base
    if (not out.is_absolute() or out.parent != base.ROOT / "results"
            or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output")
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    for i, row in enumerate(bundle["rows"]):
        base.bridge.cp5.blind.validate_row(row)
        if row["index"] != i:
            raise ValueError("row order")
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(prior.__file__).resolve(),
                 Path(prior.fixed.__file__).resolve(), base.BUNDLE):
        sources[str(path)] = base.sha(path)
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen model mismatch")
        transform = p0.image_transform()
        rows = []
        started = time.monotonic()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row in bundle["rows"]:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                initial = prior.initial_actions(row["logical_pair"])
                candidates, scores = [], []
                row_started = time.monotonic()
                for init in initial:
                    current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([current], lr=LR)
                    for _ in range(STEPS):
                        optimizer.zero_grad(set_to_none=True)
                        pred, goal = prior.fixed.replay(model, row, cp5, current)
                        loss = (pred-goal).square().mean()
                        loss.backward()
                        if current.grad is None or not bool(torch.isfinite(current.grad).all()):
                            raise ValueError("nonfinite gradient")
                        optimizer.step()
                        with torch.no_grad():
                            current.clamp_(-1, 1)
                    with torch.no_grad():
                        pred, goal = prior.fixed.replay(model, row, cp5, current)
                        scores.append(float((pred-goal).square().mean()))
                    candidates.append(current.detach()[0].cpu().numpy().copy())
                torch.cuda.synchronize(device)
                winner = prior.pick(scores)
                actions = np.asarray(candidates[winner], dtype="<f4")
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "initial_f32_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                             "candidate_scores": scores, "selected_index": winner, "selected_actions": actions.tolist(),
                             "selected_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                             "planning_wall_seconds": time.monotonic()-row_started,
                             "calls": {"action_encoder": 168, "predict": 168, "backward": 80,
                                       "post_update_candidate_scores": 4}})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model) or any(base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": before,
                  "config": {"seed": prior.SEED, "starts": prior.STARTS, "steps_per_start": STEPS,
                             "optimizer": "Adam", "learning_rate": LR, "betas": [0.9, 0.999], "eps": 1e-8,
                             "initialization": "same_as_multistart20", "action_box": [-1, 1]},
                  "encode_calls": 24, "elapsed_excluding_load": time.monotonic()-started,
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Known roots, post-outcome design; call counts from code, not independent hooks; "
                                 "different optimizer cost and no total budget certification; no physical execution."}
        digest = base.write(out, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Run fixed Adam vs SGD suffix optimization with identical starts and update count.", flush=True)
    run(Path(parser.parse_args().output))
