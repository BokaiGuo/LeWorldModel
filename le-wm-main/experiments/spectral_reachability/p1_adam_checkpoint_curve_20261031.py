"""Save fixed four-start Adam checkpoints for later physical budget comparisons."""
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

NAME = "p1_adam_checkpoint_curve_20261031"
CHECKPOINTS = (5, 10, 20)
PROTOCOL = planner.bundle.ROOT.parent / "docs/D2P_P1_ADAM_CHECKPOINT_CURVE_PROTOCOL_20261031_MAIN.md"


def pack_checkpoint(actions, scores, pick):
    if len(actions) != 4 or len(scores) != 4 or not np.isfinite(scores).all():
        raise ValueError("four finite candidates required")
    candidates = [np.asarray(action, dtype="<f4") for action in actions]
    if any(a.shape != (2, 5, 2) or not np.isfinite(a).all() or (np.abs(a) > 1).any()
           for a in candidates):
        raise ValueError("invalid raw actions")
    winner = pick(scores)
    return {"scores": scores, "winner": winner,
            "candidate_actions": [a.tolist() for a in candidates],
            "candidate_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in candidates]}


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    planner.configure()
    adam, prior = planner.adam, planner.adam.prior
    base = prior.base
    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(adam.__file__).resolve(),
                 Path(prior.__file__).resolve(), Path(prior.fixed.__file__).resolve(),
                 base.BUNDLE, PROTOCOL):
        sources[str(path)] = base.sha(path)
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
            for index, row in enumerate(bundle["rows"]):
                base.bridge.cp5.blind.validate_row(row)
                if row["index"] != index:
                    raise ValueError("row identity")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                initial = prior.initial_actions(row["logical_pair"])
                snapshots = {step: {"actions": [], "scores": []} for step in CHECKPOINTS}
                for init in initial:
                    current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([current], lr=adam.LR)
                    for step in range(1, 21):
                        optimizer.zero_grad(set_to_none=True)
                        pred, goal = prior.fixed.replay(model, row, cp5, current)
                        (pred - goal).square().mean().backward()
                        if current.grad is None or not bool(torch.isfinite(current.grad).all()):
                            raise ValueError("nonfinite gradient")
                        optimizer.step()
                        with torch.no_grad():
                            current.clamp_(-1, 1)
                            if step in CHECKPOINTS:
                                pred, goal = prior.fixed.replay(model, row, cp5, current)
                                snapshots[step]["scores"].append(float((pred-goal).square().mean()))
                                snapshots[step]["actions"].append(
                                    np.asarray(current.detach()[0].cpu().numpy(), dtype="<f4").copy())
                records = {str(step): pack_checkpoint(snapshots[step]["actions"],
                                                       snapshots[step]["scores"], prior.pick)
                           for step in CHECKPOINTS}
                rows.append({"index": index, "logical_pair": row["logical_pair"], "role": row["role"],
                             "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                             "checkpoints": records})
                print(f"completed {index + 1}/24", flush=True)
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": after,
                  "calls_per_row_code_path": {"encode": 1, "predict": 184,
                                               "action_encoder": 184, "backward": 80},
                  "checkpoints": list(CHECKPOINTS), "elapsed_seconds": time.monotonic()-started,
                  "independent_confirmation": False, "P1_admission": False,
                  "equal_budget_certified": False, "physical_execution": False,
                  "limitations": "Known roots; intermediate scoring overhead; code-path counts, "
                                 "not full cost or independent physical evidence."}
        digest = base.write(output, result)
    print(json.dumps({"output": str(output), "sha256": digest, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Save 5/10/20-update model candidates for later physical testing; no outcome selection.", flush=True)
    run(parser.parse_args().output)
