"""Compare the deployed two-packet replay to the installed LeWM rollout on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_vector_replay_20261030_main as manual
from . import codex_p1_adam_paired_join_20261030_main as sources
from . import codex_p1_adam_cp5_20261030_main as adam

base = adam.prior.base
ROOT = base.ROOT
NAME = "codex_p1_native_rollout_equivalence_20261030_main"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    need(out.is_absolute() and out.parent == ROOT / "results"
         and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(), "exclusive output")
    bundle, _ = adam.prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    plan_path, plan_sha = sources.SOURCES["plan"]
    plan, _ = adam.prior.gap.read(ROOT / plan_path, plan_sha)
    need(len(bundle["rows"]) == len(plan["rows"]) == 24, "denominator")
    deps = base.bindings()
    for path in (Path(__file__).resolve(), Path(manual.__file__).resolve(), base.BUNDLE):
        deps[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "checkpoint")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, fixed in zip(bundle["rows"], plan["rows"], strict=True):
                need(row["index"] == fixed["index"] and row["logical_pair"] == fixed["logical_pair"]
                     and row["role"] == fixed["role"], "row identity")
                raw = np.asarray(fixed["selected_actions"], dtype="<f4")
                need(raw.shape == (2, 5, 2) and
                     hashlib.sha256(raw.tobytes()).hexdigest() == fixed["selected_action_sha256"],
                     "action hash")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                need(hashlib.sha256(pixels.tobytes()).hexdigest() == row["cp5_pixel_sha256"], "cp5 pixels")
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform,
                                                                   device=device)})["emb"][0, 0]
                action = torch.tensor(raw[None], device=device, requires_grad=True)
                predicted, goal = manual.replay(model, row, cp5, action)
                loss = (predicted - goal).square().mean()
                gradient, = torch.autograd.grad(loss, action)
                other = torch.tensor(raw[None], device=device, requires_grad=True)
                states, history, goal2, mean, std = manual.base.bridge.cp5.context(row, cp5, device=device)
                future = ((other - mean) / std).reshape(1, 2, 10)
                all_actions = torch.cat((history, future), dim=1)[:, None]
                placeholder = torch.empty((1, 1, 3, 1), device=device)
                output = model.rollout({"pixels": placeholder, "emb": states[:, None]}, all_actions,
                                       history_size=3)["predicted_emb"][0, 0, -1]
                native_loss = (output - goal2[0]).square().mean()
                native_grad, = torch.autograd.grad(native_loss, other)
                need(bool(torch.isfinite(output).all() and torch.isfinite(native_grad).all()), "finite")
                a = predicted.detach().cpu().numpy()
                b = output.detach().cpu().numpy()
                ga = gradient.detach().cpu().numpy().reshape(-1)
                gb = native_grad.detach().cpu().numpy().reshape(-1)
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "action_sha256": fixed["selected_action_sha256"],
                             "manual_latent": a.tolist(), "native_latent": b.tolist(),
                             "manual_gradient": ga.tolist(), "native_gradient": gb.tolist(),
                             "manual_score": float(loss.detach()), "native_score": float(native_loss.detach()),
                             "latent_max_abs": float(np.max(np.abs(a-b))),
                             "gradient_max_abs": float(np.max(np.abs(ga-gb)))})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(base.sha(Path(p)) == h for p, h in deps.items()),
             "model or sources changed")
    result = {"schema": NAME, "status": "complete", "rows": rows, "host": host,
              "model_before": before, "model_after": before, "sources": deps, "plan_sha256": plan_sha,
              "summary": {"max_latent_abs": max(x["latent_max_abs"] for x in rows),
                          "max_gradient_abs": max(x["gradient_max_abs"] for x in rows)},
              "independent_confirmation": False, "P1_admission": False,
              "limitations": ["known roots", "same installed model and preprocessing",
                              "no physical execution or complete source closure"]}
    print(json.dumps({"sha256": base.write(out, result), "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    print("Compare fixed two-packet replay with installed LeWM rollout and gradients on CUDA.", flush=True)
    run(Path(ap.parse_args().output))
