"""Fixed 24-row real-midpoint packet replanning on the frozen CUDA model."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_multistart_rgb_20261030_main as rgb
from . import codex_p1_same_first5_feedback_compare_20261030_main as traces
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback

base = rgb.base
NAME = "p1_uphill_midpoint_feedback_plan_20261031_main"
PROTOCOL = base.ROOT.parent / "docs/D2P_P1_UPHILL_MIDPOINT_FEEDBACK_PROTOCOL_20261031_MAIN.md"


def inputs():
    pairs = rgb.paired.read(rgb.PAIR, rgb.PAIR_SHA)
    saved = rgb.paired.read(rgb.paired.CONSUME, rgb.paired.CONSUME_SHA)
    plan = rgb.paired.read(rgb.paired.physical.PLAN, rgb.paired.physical.PLAN_SHA)
    bundle = rgb.paired.read(rgb.paired.physical.planner.bundle.OUT, rgb.paired.physical.planner.BUNDLE_SHA)
    report = rgb.paired.read(rgb.RUN / "run_report.json", rgb.REPORT_SHA)
    if any(len(x) != 24 for x in (pairs["rows"], saved["rows"], plan["rows"], bundle["rows"], report["shards"])):
        raise ValueError("fixed roster")
    sources = {str(rgb.PAIR): rgb.PAIR_SHA, str(rgb.paired.CONSUME): rgb.paired.CONSUME_SHA,
               str(rgb.paired.physical.PLAN): rgb.paired.physical.PLAN_SHA,
               str(rgb.paired.physical.planner.bundle.OUT): rgb.paired.physical.planner.BUNDLE_SHA,
               str(rgb.RUN / "run_report.json"): rgb.REPORT_SHA}
    entries = []
    for i, (pair, observed, proposal, row, shard) in enumerate(zip(
            pairs["rows"], saved["rows"], plan["rows"], bundle["rows"], report["shards"], strict=True)):
        if not (pair["index"] == observed["index"] == proposal["index"] == row["index"] == shard["index"] == i
                and pair["role"] == observed["role"] == proposal["role"] == row["role"]
                and observed["returned_suffix"] >= 5):
            raise ValueError("row identity or early stop")
        raw = np.asarray(proposal["selected_actions"], dtype="<f4")
        if raw.shape != (2, 5, 2) or hashlib.sha256(raw.tobytes()).hexdigest() != proposal["selected_action_sha256"]:
            raise ValueError("action identity")
        folder = rgb.RUN / f"row_{i:02d}"
        events = traces.transitions(folder, shard["report_sha256"])
        if len(events) < 5 or any(t["step"] != j+1 or t["actual_bits"] != raw[0, j].tobytes().hex()
                                  for j, t in enumerate(events[:5])):
            raise ValueError("executed first packet")
        row_report = rgb.paired.read(folder / "report.json", shard["report_sha256"])
        frames = [f for f in row_report["row"]["trajectory"] if f["phase"] == "probe" and f["step"] == 5
                  and f["condition"] == observed["condition"]
                  and f["overlay_index"] == (1 if observed["condition"] == "on" else 0)]
        if len(frames) != 1 or frames[0]["state"] != events[4]["state"]:
            raise ValueError("midpoint binding")
        frame = frames[0]
        path = folder / frame["png"]
        if path.name != frame["png"] or base.sha(path) != frame["file_sha256"]:
            raise ValueError("midpoint image")
        with Image.open(path) as image:
            image_array = np.asarray(image).copy()
        if (image_array.shape != (224, 224, 3) or image_array.dtype != np.uint8
                or hashlib.sha256(image_array.tobytes()).hexdigest() != frame["pixel_sha256"]):
            raise ValueError("midpoint pixels")
        sources[str(folder / "report.json")] = shard["report_sha256"]
        sources[str(folder / "journal.jsonl")] = row_report["journal_sha256"]
        sources[str(path)] = frame["file_sha256"]
        entries.append((row, pair, raw, image_array, frame))
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("output")
    entries, sources = inputs()
    sources.update(base.bindings())
    for path in (Path(__file__).resolve(), Path(feedback.__file__).resolve(), PROTOCOL):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, pair, raw, midpoint_rgb, frame in entries:
                base.bridge.cp5.blind.validate_row(row)
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_rgb = np.asarray(image).copy()
                with torch.no_grad():
                    encode = lambda im: model.encode({"pixels": p0._image_tensor(
                        im, transform=transform, device=device)})["emb"][0, 0]
                    cp, midpoint = encode(cp_rgb), encode(midpoint_rgb)
                    states, history, goal, mean, std = feedback.second_context(row, cp, midpoint, raw[0])
                action = torch.nn.Parameter(torch.tensor(raw[1][None], dtype=torch.float32, device=device))
                optimizer = torch.optim.Adam([action], lr=0.05)

                def score():
                    token = ((action - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                    return (pred-goal).square().mean()

                torch.cuda.synchronize(device)
                start = time.monotonic()
                with torch.no_grad():
                    baseline = float(score())
                for _ in range(20):
                    optimizer.zero_grad(set_to_none=True)
                    loss = score()
                    loss.backward()
                    if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                        raise ValueError("gradient")
                    optimizer.step()
                    with torch.no_grad():
                        action.clamp_(-1, 1)
                with torch.no_grad():
                    final = float(score())
                if not np.isfinite([baseline, final]).all():
                    raise ValueError("nonfinite score")
                torch.cuda.synchronize(device)
                chosen = action.detach().cpu().numpy()[0].astype("<f4") if final < baseline else raw[1].copy()
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "first_packet_hex": raw[0].tobytes().hex(), "second_packet_hex": chosen.tobytes().hex(),
                             "midpoint_pixel_sha256": frame["pixel_sha256"], "baseline_score": baseline,
                             "optimized_score": final, "chosen": "optimized" if final < baseline else "original",
                             "optimizer_seconds": time.monotonic()-start})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": after,
                  "per_row_calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20},
                  "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
                  "limitations": "Known roots, post-outcome development, offline midpoint, extra optimization."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replan all fixed final packets from real midpoints on host CUDA.", flush=True)
    run(args.output)
