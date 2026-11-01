"""Independently remeasure 10/20-update frozen-model suffix planning on CUDA."""
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
from . import p1_adam_checkpoint_curve_20261031 as curve
from . import p1_uphill_multistart_physical_20261030_main as physical

ROOT = planner.bundle.ROOT
NAME = "p1_accounted_adam_replay_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_ACCOUNTED_ADAM_REPLAY_PROTOCOL_20261031.md"
CURVE = ROOT / "results/p1_adam_checkpoint_curve_20261031_RUN01.json"
CURVE_SHA = "46d7126e0c0f5354f829fa0acac27e1639c1f8ef94ac74ebcca21cfda55ef2a7"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    planner.configure()
    prior = planner.adam.prior
    base = prior.base
    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" or not output.name.startswith(NAME + "_") \
            or output.exists() or output.is_symlink():
        raise ValueError("exclusive output")
    if base.sha(CURVE) != CURVE_SHA or base.sha(physical.PLAN) != physical.PLAN_SHA:
        raise ValueError("history hash")
    historical = json.loads(CURVE.read_bytes())
    old_plan = json.loads(physical.PLAN.read_bytes())
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if not len(bundle["rows"]) == len(historical["rows"]) == len(old_plan["rows"]) == 24:
        raise ValueError("fixed roster")
    sources = base.bindings()
    for p in (Path(__file__).resolve(), PROTOCOL, CURVE, physical.PLAN,
              Path(prior.__file__).resolve(), Path(prior.fixed.__file__).resolve(), base.BUNDLE):
        sources[str(p)] = base.sha(p)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen checkpoint")
        counts = {"encode": 0, "predict": 0, "action_encoder": 0, "backward": 0}
        original_encode, original_predict = model.encode, model.predict

        def encode(*args, **kwargs):
            counts["encode"] += 1
            return original_encode(*args, **kwargs)

        def predict(*args, **kwargs):
            counts["predict"] += 1
            return original_predict(*args, **kwargs)

        model.encode, model.predict = encode, predict
        hook = model.action_encoder.register_forward_hook(
            lambda module, inputs, output: counts.__setitem__("action_encoder", counts["action_encoder"] + 1))
        transform = p0.image_transform()
        try:
            with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                for steps in (10, 20):
                    for i, row in enumerate(bundle["rows"]):
                        base.bridge.cp5.blind.validate_row(row)
                        if row["index"] != i or historical["rows"][i]["index"] != i or old_plan["rows"][i]["index"] != i:
                            raise ValueError("row identity")
                        for key in counts:
                            counts[key] = 0
                        torch.cuda.synchronize(device)
                        started = time.monotonic()
                        with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                            pixels = np.asarray(image).copy()
                        with torch.no_grad():
                            cp = model.encode({"pixels": p0._image_tensor(
                                pixels, transform=transform, device=device)})["emb"][0, 0]
                        starts = prior.initial_actions(row["logical_pair"])
                        candidates, scores = [], []
                        for initial in starts:
                            action = torch.nn.Parameter(torch.tensor(initial[None], dtype=torch.float32, device=device))
                            optimizer = torch.optim.Adam([action], lr=planner.adam.LR)
                            for _ in range(steps):
                                optimizer.zero_grad(set_to_none=True)
                                pred, goal = prior.fixed.replay(model, row, cp, action)
                                (pred - goal).square().mean().backward()
                                counts["backward"] += 1
                                if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                                    raise ValueError("gradient")
                                optimizer.step()
                                with torch.no_grad():
                                    action.clamp_(-1, 1)
                            with torch.no_grad():
                                pred, goal = prior.fixed.replay(model, row, cp, action)
                                scores.append(float((pred - goal).square().mean()))
                            candidates.append(np.asarray(action.detach()[0].cpu().numpy(), dtype="<f4").copy())
                        torch.cuda.synchronize(device)
                        elapsed = time.monotonic() - started
                        winner = prior.pick(scores)
                        old = historical["rows"][i]["checkpoints"][str(steps)]
                        reference = [np.asarray(a, dtype="<f4") for a in old["candidate_actions"]]
                        if len(reference) != 4 or any(a.shape != (2, 5, 2) for a in reference):
                            raise ValueError("reference candidates")
                        max_difference = max(float(np.max(np.abs(a - b))) for a, b in zip(candidates, reference, strict=True))
                        hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in candidates]
                        rows.append({"index": i, "logical_pair": row["logical_pair"], "role": row["role"],
                                     "updates_per_start": steps, "counts": dict(counts),
                                     "synchronized_wall_seconds_including_rgb_decode": elapsed,
                                     "scores": scores, "winner": winner, "candidate_sha256": hashes,
                                     "all_candidates_byte_identical": hashes == old["candidate_sha256"],
                                     "max_action_abs_difference": max_difference,
                                     "winner_matches_curve": winner == old["winner"],
                                     "winner_matches_original20": (winner == old_plan["rows"][i]["selected_index"]
                                         and hashes[winner] == old_plan["rows"][i]["selected_action_sha256"])
                                     if steps == 20 else None})
                        print(json.dumps({"steps": steps, "index": i, "counts": counts,
                                          "identical": rows[-1]["all_candidates_byte_identical"]}), flush=True)
        finally:
            hook.remove()
            model.encode, model.predict = original_encode, original_predict
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": after,
                  "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
                  "limitations": "Known roots; 10 then 20 fixed order; excludes model loading, prefix generation, "
                                 "physics and earlier selection; not an end-to-end cost or online feedback."}
        digest = base.write(output, result)
    print(json.dumps({"sha256": digest, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
