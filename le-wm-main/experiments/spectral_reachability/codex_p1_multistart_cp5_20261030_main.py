"""Fixed four-start, five-update exploratory planner on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_cp5_vector_replay_20261030_main as fixed

NAME = "codex_p1_multistart_cp5_20261030_main"
SEED = 20261030
STEPS = 5
LEARNING_RATE = 0.05
STARTS = 4


def initial_actions(logical_pair):
    if type(logical_pair) is not int or not 0 <= logical_pair < 8:
        raise ValueError("logical pair")
    rng = np.random.default_rng(SEED + logical_pair)
    samples = rng.uniform(-1.0, 1.0, size=(STARTS, 2, 5, 2)).astype("<f4")
    samples[0] = 0
    return samples


def pick(scores):
    if len(scores) != STARTS or not np.isfinite(scores).all():
        raise ValueError("scores")
    return min(range(STARTS), key=lambda i: (scores[i], i))


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
    bundle, _ = gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    for i, row in enumerate(bundle["rows"]):
        base.bridge.cp5.blind.validate_row(row)
        if row["index"] != i:
            raise ValueError("row identity")
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(fixed.__file__).resolve(), base.BUNDLE):
        sources[str(path)] = base.sha(path)
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model mismatch")
        transform = p0.image_transform()
        rows = []
        started = time.monotonic()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row in bundle["rows"]:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                initial = initial_actions(row["logical_pair"])
                candidates, scores = [], []
                row_started = time.monotonic()
                for init in initial:
                    current = torch.tensor(init[None], dtype=torch.float32, device=device)
                    for _ in range(STEPS):
                        current = current.detach().requires_grad_(True)
                        pred, goal = fixed.replay(model, row, cp5, current)
                        loss = (pred-goal).square().mean()
                        gradient, = torch.autograd.grad(loss, current)
                        if not bool(torch.isfinite(gradient).all()):
                            raise ValueError("nonfinite gradient")
                        current = (current - LEARNING_RATE * gradient).clamp(-1, 1).detach()
                    with torch.no_grad():
                        pred, goal = fixed.replay(model, row, cp5, current)
                        scores.append(float((pred-goal).square().mean()))
                    candidates.append(current[0].cpu().numpy())
                torch.cuda.synchronize(device)
                winner = pick(scores)
                actions = np.asarray(candidates[winner], dtype="<f4")
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "initial_f32_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                             "candidate_scores": scores, "selected_index": winner,
                             "selected_actions": actions.tolist(),
                             "selected_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                             "planning_wall_seconds": time.monotonic()-row_started,
                             "calls": {"action_encoder": 48, "predict": 48, "backward": 20,
                                       "post_update_candidate_scores": 4}})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": before,
                  "config": {"seed": SEED, "starts": STARTS, "steps_per_start": STEPS,
                             "learning_rate": LEARNING_RATE, "first_start": "zero",
                             "other_starts": "uniform_raw_box", "same_init_within_logical_pair": True},
                  "encode_calls": 24, "elapsed_excluding_load": time.monotonic()-started,
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Known roots, duplicate trajectories, chosen after prior outcomes; "
                                 "no OS isolation; explicit call counts not independent hook ledger; "
                                 "not equal cost to 20-step baseline; no physical execution."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Run fixed four-start search on each known-root cp5 input, no physical execution.", flush=True)
    run(args.output)
