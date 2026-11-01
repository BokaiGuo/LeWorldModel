"""Explore a trajectory-aware frozen-model suffix objective on host CUDA.

This is a new known-root diagnostic.  It does not alter the historical
terminal-only results or the P1 gate.  The physical environment can stop at
an intermediate step, while the previous optimizer selected only the final
packet endpoint.  We therefore optimize a fixed soft-min over the two packet
endpoint costs and report the best endpoint separately.
"""

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_packet_objective_adam_20261031 as terminal
from . import p1_uphill_multistart_adam_20261030_main as source

NAME = "p1_trajectory_objective_adam_20261031"
PROTOCOL = source.bundle.ROOT.parent / "docs/D2P_P1_TRAJECTORY_OBJECTIVE_PROTOCOL_20261031.md"
TAU = 0.02
STEPS = 20


def trajectory_losses(model, row, cp5, raw):
    """Return differentiable packet endpoint losses in chronological order."""
    import torch

    losses = terminal.packet_losses(model, row, cp5, raw)
    if len(losses) != 2:
        raise ValueError("two packet losses required")
    if any(loss.ndim != 0 or not bool(torch.isfinite(loss).item()) for loss in losses):
        raise ValueError("finite scalar packet losses required")
    return losses


def softmin(losses, tau=TAU):
    import torch

    if len(losses) != 2 or tau <= 0:
        raise ValueError("softmin contract")
    values = torch.stack(tuple(losses))
    return -tau * torch.logsumexp(-values / tau, dim=0)


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base, prior = source.adam.prior.base, source.adam.prior
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    if base.sha(source.bundle.OUT) != source.BUNDLE_SHA:
        raise ValueError("bundle hash")
    bundle, _ = prior.gap.read(source.bundle.OUT, source.BUNDLE_SHA)
    if len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    for i, row in enumerate(bundle["rows"]):
        base.bridge.cp5.blind.validate_row(row)
        if row["index"] != i or row["logical_pair"] != i // 3 \
                or row["role"] != ("selected", "zero", "greedy")[i % 3]:
            raise ValueError("roster identity")

    bindings = base.bindings()
    for path in (Path(__file__).resolve(), PROTOCOL, source.bundle.OUT,
                 Path(terminal.__file__).resolve(), Path(prior.__file__).resolve(),
                 Path(prior.fixed.__file__).resolve()):
        bindings[str(path)] = base.sha(path)

    rows = []
    started = time.monotonic()
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen checkpoint")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row in bundle["rows"]:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image.convert("RGB")).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                initial = prior.initial_actions(row["logical_pair"])
                candidates, objectives, endpoint_losses, best_packet = [], [], [], []
                row_started = time.monotonic()
                for init in initial:
                    current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([current], lr=source.adam.LR)
                    for _ in range(STEPS):
                        optimizer.zero_grad(set_to_none=True)
                        losses = trajectory_losses(model, row, cp5, current)
                        objective = softmin(losses)
                        objective.backward()
                        if current.grad is None or not bool(torch.isfinite(current.grad).all()):
                            raise ValueError("nonfinite gradient")
                        optimizer.step()
                        with torch.no_grad():
                            current.clamp_(-1, 1)
                    with torch.no_grad():
                        losses = trajectory_losses(model, row, cp5, current)
                        values = [float(x.item()) for x in losses]
                        objective = float(softmin(losses).item())
                    action = np.asarray(current.detach()[0].cpu().numpy(), dtype="<f4").copy()
                    candidates.append(action)
                    objectives.append(objective)
                    endpoint_losses.append(values)
                    best_packet.append(int(np.argmin(values)))
                winner = prior.pick(objectives)
                actions = candidates[winner]
                rows.append({
                    "index": row["index"],
                    "logical_pair": row["logical_pair"],
                    "role": row["role"],
                    "initial_f32_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                    "candidate_objectives": objectives,
                    "candidate_packet_endpoint_losses": endpoint_losses,
                    "candidate_best_packet": best_packet,
                    "selected_index": winner,
                    "selected_actions": actions.tolist(),
                    "selected_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                    "planning_wall_seconds": time.monotonic() - row_started,
                    "objective": "softmin([packet5_goal_mse, packet10_goal_mse], tau=0.02)",
                    "calls": {
                        "action_encoder": 168,
                        "predict": 168,
                        "backward": 80,
                        "post_update_candidate_scores": 4,
                    },
                })
                print(json.dumps({"index": row["index"], "selected_index": winner,
                                  "best_packet": best_packet[winner]}), flush=True)
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(Path(path)) != digest for path, digest in bindings.items()):
            raise ValueError("model or source drift")
        result = {
            "schema": NAME,
            "status": "complete",
            "rows": rows,
            "sources": bindings,
            "host": host,
            "model_before": before,
            "model_after": after,
            "config": {
                "seed": prior.SEED,
                "starts": 4,
                "steps_per_start": STEPS,
                "optimizer": "Adam",
                "learning_rate": source.adam.LR,
                "action_box": [-1, 1],
                "softmin_tau": TAU,
            },
            "elapsed_excluding_load": time.monotonic() - started,
            "P1_admission": False,
            "independent_confirmation": False,
            "equal_budget_certified": False,
            "physical_execution": False,
            "limitations": "Known-root diagnostic only; no outcome input, physical execution, "
                            "independent confirmation, or end-to-end equal-budget claim.",
        }
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Plan fixed RGB prefixes with a trajectory-aware soft-min endpoint objective on host CUDA.",
          flush=True)
    run(Path(parser.parse_args().output))
