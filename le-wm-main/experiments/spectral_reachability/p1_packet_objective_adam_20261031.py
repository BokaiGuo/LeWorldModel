"""Fixed two-packet objective on the known-root uphill cp5 RGB bundle."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_multistart_adam_20261030_main as source

NAME = "p1_packet_objective_adam_20261031"
PROTOCOL = source.bundle.ROOT.parent / "docs/D2P_P1_PACKET_OBJECTIVE_PROTOCOL_20261031.md"


def packet_losses(model, row, cp5, raw):
    import torch
    base = source.adam.prior.base
    states, history, goal, mean, std = base.bridge.cp5.context(row, cp5, device=cp5.device)
    losses = []
    for packet in range(2):
        norm = ((raw[:, packet] - mean) / std).reshape(1, 1, 10)
        context = torch.cat((history, norm), dim=1)
        prediction = model.predict(states, model.action_encoder(context))
        if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError("prediction")
        losses.append((prediction[:, -1] - goal).square().mean())
        states = torch.cat((states, prediction[:, -1:]), dim=1)[:, -3:]
        history = context[:, -2:]
    return losses


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base, prior = source.adam.prior.base, source.adam.prior
    if (not out.is_absolute() or out.parent != base.ROOT / "results"
            or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output")
    if base.sha(source.bundle.OUT) != source.BUNDLE_SHA:
        raise ValueError("bundle hash")
    bundle, _ = prior.gap.read(source.bundle.OUT, source.BUNDLE_SHA)
    if len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    for i, row in enumerate(bundle["rows"]):
        base.bridge.cp5.blind.validate_row(row)
        if row["index"] != i or row["logical_pair"] != i // 3 or row["role"] != ("selected", "zero", "greedy")[i % 3]:
            raise ValueError("roster identity")
    bindings = base.bindings()
    for p in (Path(__file__).resolve(), PROTOCOL, source.bundle.OUT,
              Path(prior.__file__).resolve(), Path(prior.fixed.__file__).resolve()):
        bindings[str(p)] = base.sha(p)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen checkpoint")
        transform = p0.image_transform()
        started = time.monotonic()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row in bundle["rows"]:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as im:
                    pixels = np.asarray(im.convert("RGB")).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                initial = prior.initial_actions(row["logical_pair"])
                candidates, scores, components = [], [], []
                row_started = time.monotonic()
                for init in initial:
                    current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([current], lr=source.adam.LR)
                    for _ in range(source.adam.STEPS):
                        optimizer.zero_grad(set_to_none=True)
                        a, b = packet_losses(model, row, cp5, current)
                        (0.5 * (a + b)).backward()
                        if current.grad is None or not bool(torch.isfinite(current.grad).all()):
                            raise ValueError("nonfinite gradient")
                        optimizer.step()
                        with torch.no_grad():
                            current.clamp_(-1, 1)
                    with torch.no_grad():
                        a, b = packet_losses(model, row, cp5, current)
                        components.append([float(a), float(b)])
                        scores.append(float(0.5 * (a + b)))
                    candidates.append(current.detach()[0].cpu().numpy().copy())
                torch.cuda.synchronize(device)
                winner = prior.pick(scores)
                actions = np.asarray(candidates[winner], dtype="<f4")
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "initial_f32_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                             "candidate_scores": scores, "candidate_packet_scores": components,
                             "selected_index": winner, "selected_actions": actions.tolist(),
                             "selected_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                             "planning_wall_seconds": time.monotonic() - row_started,
                             "calls": {"action_encoder": 168, "predict": 168, "backward": 80,
                                       "post_update_candidate_scores": 4}})
                print(json.dumps({"index": row["index"], "selected_index": winner}), flush=True)
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in bindings.items()):
            raise ValueError("model or source drift")
        value = {"schema": NAME, "status": "complete", "rows": rows, "sources": bindings,
                 "host": host, "model_before": before, "model_after": after,
                 "config": {"seed": prior.SEED, "starts": 4, "steps_per_start": 20,
                            "optimizer": "Adam", "learning_rate": 0.05, "action_box": [-1, 1],
                            "objective": "0.5 * (goal_mse_at_5 + goal_mse_at_10)"},
                 "encode_calls": 24, "elapsed_excluding_load": time.monotonic() - started,
                 "P1_admission": False, "independent_confirmation": False,
                 "limitations": "Post-outcome known-root development; no physical execution; "
                                "call counts from code path, not certified equal total budget."}
        digest = base.write(out, value)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Plan all fixed RGB prefixes with equal-weight packet and terminal latent costs on CUDA.", flush=True)
    run(Path(parser.parse_args().output))
