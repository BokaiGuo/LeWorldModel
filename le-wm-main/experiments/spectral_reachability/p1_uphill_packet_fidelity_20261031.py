"""Compare fixed action-packet predictions with same-time physical RGB on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_multistart_rgb_20261030_main as prior
from . import codex_p1_cp5_vector_replay_20261030_main as replay

base = prior.base
NAME = "p1_uphill_packet_fidelity_20261031"
PLAN = base.ROOT / "results/p1_uphill_multistart_adam_20261030_main_RUN01.json"
PLAN_SHA = "018dffcb53e5e4dd40e6323fc3665ca53321cb027023c0cc4c1edb5daa06229f"


def frame_at(row, step, condition):
    frames = [f for f in row["trajectory"] if f["phase"] == "probe"
              and f["step"] == step and f["overlay_index"] == (condition == "on")]
    if (len(frames) != 1 or len(row["responses"]) < step
            or frames[0]["state"] != row["responses"][step - 1]["state"]
            or frames[0]["condition"] != condition):
        raise ValueError("same-time RGB and physical response binding")
    return frames[0]


def inputs():
    paired = prior.paired
    pairs = paired.read(prior.PAIR, prior.PAIR_SHA)
    report = paired.read(prior.RUN / "run_report.json", prior.REPORT_SHA)
    bundle = paired.read(paired.physical.planner.bundle.OUT, paired.physical.planner.BUNDLE_SHA)
    observed = paired.read(paired.CONSUME, paired.CONSUME_SHA)
    plan = paired.read(PLAN, PLAN_SHA)
    if not (len(pairs["rows"]) == len(report["shards"]) == len(bundle["rows"])
            == len(observed["rows"]) == len(plan["rows"]) == 24
            and report["status"] == "recorded_not_consumed"):
        raise ValueError("fixed roster")
    sources = {str(p): digest for p, digest in (
        (prior.PAIR, prior.PAIR_SHA), (prior.RUN / "run_report.json", prior.REPORT_SHA),
        (paired.physical.planner.bundle.OUT, paired.physical.planner.BUNDLE_SHA),
        (paired.CONSUME, paired.CONSUME_SHA), (PLAN, PLAN_SHA))}
    entries = []
    for i, (pair, shard, blind, physical, proposal) in enumerate(zip(
            pairs["rows"], report["shards"], bundle["rows"], observed["rows"],
            plan["rows"], strict=True)):
        if not (i == pair["index"] == shard["index"] == blind["index"]
                == physical["index"] == proposal["index"]
                and pair["role"] == blind["role"] == physical["role"] == proposal["role"]
                and pair["status"] == physical["status"]):
            raise ValueError("row identity")
        path = prior.RUN / f"row_{i:02d}" / "report.json"
        saved = paired.read(path, shard["report_sha256"])
        row = saved["row"]
        raw = np.asarray(proposal["selected_actions"], dtype="<f4")
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if not (raw.shape == (2, 5, 2) and np.isfinite(raw).all()
                and digest == proposal["selected_action_sha256"] == saved["raw10_sha256"]
                and row["status"] == physical["status"]
                and row["own_returned"] == 5 + physical["returned_suffix"]
                and len(row["responses"]) == physical["returned_suffix"]
                and all(r["step"] == n for n, r in enumerate(row["responses"], 1))):
            raise ValueError("action or response identity")
        sources[str(path)] = shard["report_sha256"]
        images = {}
        for step in (5, 10):
            if len(row["responses"]) < step:
                continue
            frame = frame_at(row, step, pair["condition"])
            png = path.parent / frame["png"]
            if png.parent != path.parent or base.sha(png) != frame["file_sha256"]:
                raise ValueError("image file binding")
            with Image.open(png) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            if (pixels.shape != (224, 224, 3)
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != frame["pixel_sha256"]):
                raise ValueError("image pixel binding")
            sources[str(png)] = frame["file_sha256"]
            images[step] = pixels
        if 5 not in images:
            raise ValueError("missing fixed packet observation")
        entries.append({"index": i, "pair": pair["logical_pair"], "role": pair["role"],
                        "condition": pair["condition"], "status": pair["status"],
                        "steps": len(row["responses"]), "blind": blind, "raw": raw,
                        "images": images, "score": pair["model_best_score"],
                        "native_success": pair["last_native_success"], "action_sha256": digest})
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if (not out.is_absolute() or out.parent != base.ROOT / "results"
            or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output")
    entries, sources = inputs()
    sources.update(base.bindings())
    for path in (Path(__file__).resolve(), Path(replay.__file__).resolve()):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen model")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for e in entries:
                with Image.open(io.BytesIO(base64.b64decode(e["blind"]["cp5_png_base64"], validate=True))) as im:
                    cp_pixels = np.asarray(im.convert("RGB")).copy()
                def encode(pixels):
                    return model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0]
                cp = encode(cp_pixels)
                states, history, goal, mean, std = base.bridge.cp5.context(e["blind"], cp, device=device)
                raw = torch.as_tensor(e["raw"], device=device)
                packet_metrics = []
                for packet in range(2):
                    token = ((raw[packet] - mean) / std).reshape(1, 1, 10)
                    context = torch.cat((history, token), dim=1)
                    prediction = model.predict(states, model.action_encoder(context))
                    if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError("prediction")
                    predicted = prediction[0, -1]
                    step = 5 * (packet + 1)
                    actual = encode(e["images"][step]) if step in e["images"] else None
                    packet_metrics.append({
                        "step": step, "observed": actual is not None,
                        "predicted_goal_mse": float((predicted - goal[0]).square().mean()),
                        "actual_goal_mse": None if actual is None else float((actual - goal[0]).square().mean()),
                        "predicted_actual_mse": None if actual is None else float((predicted - actual).square().mean())})
                    states = torch.cat((states, prediction[:, -1:]), dim=1)[:, -3:]
                    history = context[:, -2:]
                if abs(packet_metrics[1]["predicted_goal_mse"] - e["score"]) > 1e-5:
                    raise ValueError("planner score replay")
                rows.append({k: e[k] for k in ("index", "pair", "role", "condition",
                            "status", "steps", "native_success", "action_sha256")}
                            | {"packets": packet_metrics})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        value = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                 "host": host, "model_before": before, "model_after": before,
                 "P1_admission": False, "independent_confirmation": False,
                 "limitations": "Known roots and duplicate trajectories; observational replay; "
                                "no observations imputed after physical stop; not a policy or outcome gate."}
        digest = base.write(out, value)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare fixed action packet predictions with same-time physical RGB.", flush=True)
    run(parser.parse_args().output)
