"""Reproduce fixed Adam candidates and compare with the sealed exploratory run."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_multistart_adam_20261030_main as planner

NAME = "p1_uphill_adam_candidate_replay_20261031"
SOURCE_SHA = "018dffcb53e5e4dd40e6323fc3665ca53321cb027023c0cc4c1edb5daa06229f"
SOURCE = planner.bundle.ROOT / "results/p1_uphill_multistart_adam_20261030_main_RUN01.json"


def comparison(scores, winner, actions, saved):
    """Report differences rather than treating numerical divergence as a theory result."""
    saved_scores = np.asarray(saved["candidate_scores"], dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (4,) or saved_scores.shape != (4,) or not (
            np.isfinite(scores).all() and np.isfinite(saved_scores).all()):
        raise ValueError("four finite scores required")
    action = np.asarray(actions[winner], dtype="<f4")
    old_action = np.asarray(saved["selected_actions"], dtype="<f4")
    if action.shape != (2, 5, 2) or old_action.shape != action.shape:
        raise ValueError("action shape")
    return {
        "score_delta": (scores - saved_scores).tolist(),
        "max_abs_score_delta": float(np.max(np.abs(scores - saved_scores))),
        "same_winner": winner == saved["selected_index"],
        "same_winner_action_bytes": bool(np.array_equal(action.view("u1"), old_action.view("u1"))),
        "max_abs_winner_action_delta": float(np.max(np.abs(action - old_action))),
    }


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    planner.configure()
    adam = planner.adam
    prior = adam.prior
    base = prior.base
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    original, _ = prior.gap.read(SOURCE, SOURCE_SHA)
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if original["status"] != "complete" or len(original["rows"]) != 24 or len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    # A changed dependency would prevent a meaningful reproduction.
    sources = {str(Path(__file__).resolve()): base.sha(Path(__file__).resolve())}
    for path, digest in original["sources"].items():
        if base.sha(path) != digest:
            raise ValueError("original source drift: " + path)
        sources[path] = digest
    rows = []
    started = time.monotonic()
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint mismatch")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for i, (row, saved) in enumerate(zip(bundle["rows"], original["rows"], strict=True)):
                base.bridge.cp5.blind.validate_row(row)
                if row["index"] != i or saved["index"] != i or any(
                        row[k] != saved[k] for k in ("logical_pair", "role")):
                    raise ValueError("row identity")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                initial = prior.initial_actions(row["logical_pair"])
                if hashlib.sha256(initial.tobytes()).hexdigest() != saved["initial_f32_sha256"]:
                    raise ValueError("initialization mismatch")
                actions, scores = [], []
                for init in initial:
                    current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([current], lr=adam.LR)
                    for _ in range(adam.STEPS):
                        optimizer.zero_grad(set_to_none=True)
                        pred, goal = prior.fixed.replay(model, row, cp5, current)
                        (pred - goal).square().mean().backward()
                        if current.grad is None or not bool(torch.isfinite(current.grad).all()):
                            raise ValueError("nonfinite gradient")
                        optimizer.step()
                        with torch.no_grad():
                            current.clamp_(-1, 1)
                    with torch.no_grad():
                        pred, goal = prior.fixed.replay(model, row, cp5, current)
                        scores.append(float((pred-goal).square().mean()))
                    actions.append(np.asarray(current.detach()[0].cpu().numpy(), dtype="<f4").copy())
                winner = prior.pick(scores)
                rows.append({"index": i, "logical_pair": row["logical_pair"], "role": row["role"],
                             "scores": scores, "winner": winner,
                             "candidate_actions": [a.tolist() for a in actions],
                             "candidate_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in actions],
                             "comparison": comparison(scores, winner, actions, saved)})
                print(f"replayed {i+1}/24", flush=True)
        torch.cuda.synchronize(device)
        if before != base.model_hash(model) or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "source_sha256": SOURCE_SHA,
                  "sources": sources, "host": host, "rows": rows,
                  "model_before": before, "model_after": before,
                  "elapsed_seconds": time.monotonic() - started,
                  "calls_per_row": {"encoder": 1, "action_encoder": 168, "predict": 168, "backward": 80},
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Reproduction on known inputs; candidate score replay shares model implementation; "
                                 "not new physical data or a full budget certification."}
        digest = base.write(output, result)
    print(json.dumps({"output": str(output), "sha256": digest, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Reproduce all four fixed CUDA candidates; no physical outcomes select actions.", flush=True)
    run(parser.parse_args().output)
