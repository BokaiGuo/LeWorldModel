"""Host-CUDA replay and directional finite-difference audit of multi-start search."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_multistart_cp5_20261030_main as plan

NAME = "codex_p1_multistart_gradient_audit_20261030_main"
SAVED = plan.base.ROOT / "results/codex_p1_multistart_cp5_20261030_main_RUN01.json"
SAVED_SHA = "7e14227760f52eee3d7aad232b09c22fa43436dc90ae17fc6f9c58b4256e05f0"


def direction(index, start):
    rng = np.random.default_rng(981733 + index * 4 + start)
    value = rng.normal(size=(1, 2, 5, 2)).astype("<f4")
    value /= np.linalg.norm(value)
    return value


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != plan.base.ROOT/"results"
            or not output.name.startswith(NAME+"_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    bundle, _ = plan.gap.read(plan.base.BUNDLE, plan.base.BUNDLE_SHA)
    saved, _ = plan.gap.read(SAVED, SAVED_SHA)
    if saved["status"] != "complete" or len(saved["rows"]) != 24 or len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    sources = plan.base.bindings()
    for path in (Path(__file__).resolve(), Path(plan.__file__).resolve(), SAVED, plan.base.BUNDLE):
        sources[str(path)] = plan.base.sha(path)
    with plan.base.locking.gpu_lock(plan.base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(plan.base.MODEL), device=device).eval().requires_grad_(False)
        before = plan.base.model_hash(model)
        if before != plan.base.MODEL_HASH:
            raise ValueError("model mismatch")
        transform = p0.image_transform()
        rows = []
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, expected in zip(bundle["rows"], saved["rows"], strict=True):
                if row["index"] != expected["index"]:
                    raise ValueError("row identity")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                starts = plan.initial_actions(row["logical_pair"])
                diagnostics = []
                final_actions = []
                for start, initial in enumerate(starts):
                    current = torch.tensor(initial[None], device=device)
                    v = torch.tensor(direction(row["index"], start), device=device)
                    margin = float((1-current.abs()).min())
                    epsilon = min(1e-3, margin / 4)
                    if epsilon <= 1e-7:
                        raise ValueError("finite difference too close to boundary")
                    losses, clipped = [], []
                    for step in range(plan.STEPS):
                        current = current.detach().requires_grad_(True)
                        pred, goal = plan.fixed.replay(model, row, cp5, current)
                        loss = (pred-goal).square().mean()
                        gradient, = torch.autograd.grad(loss, current)
                        if not bool(torch.isfinite(gradient).all()):
                            raise ValueError("nonfinite gradient")
                        if step == 0:
                            with torch.no_grad():
                                plus, _ = plan.fixed.replay(model, row, cp5, current + epsilon*v)
                                minus, _ = plan.fixed.replay(model, row, cp5, current - epsilon*v)
                                fd = float(((plus-goal).square().mean()-(minus-goal).square().mean())/(2*epsilon))
                            ad = float((gradient*v).sum())
                        losses.append(float(loss.detach()))
                        update = current - plan.LEARNING_RATE*gradient
                        clipped.append(int((update.abs() > 1).sum()))
                        current = update.clamp(-1, 1).detach()
                    with torch.no_grad():
                        pred, goal = plan.fixed.replay(model, row, cp5, current)
                        losses.append(float((pred-goal).square().mean()))
                    final_actions.append(current[0].cpu().numpy())
                    diagnostics.append({"start": start, "epsilon": epsilon, "ad_directional": ad,
                                        "fd_directional": fd, "absolute_error": abs(ad-fd),
                                        "losses": losses, "clipped_coordinates": clipped})
                scores = [x["losses"][-1] for x in diagnostics]
                winner = plan.pick(scores)
                actions = np.asarray(final_actions[winner], dtype="<f4")
                if (winner != expected["selected_index"] or
                        max(abs(a-b) for a, b in zip(scores, expected["candidate_scores"], strict=True)) > 1e-5 or
                        hashlib.sha256(actions.tobytes()).hexdigest() != expected["selected_action_sha256"]):
                    raise ValueError("saved winner or score mismatch")
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "selected_index": winner, "starts": diagnostics})
        torch.cuda.synchronize(device)
        if plan.base.model_hash(model) != before or any(plan.base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": before, "P1_admission": False,
                  "limitations": "Directional finite difference at initial actions only, not a full Jacobian "
                                 "audit or physical control comparison; known roots and duplicate trajectories."}
        digest = plan.base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Audit same-start search replay and bounded directional finite differences.", flush=True)
    run(args.output)
