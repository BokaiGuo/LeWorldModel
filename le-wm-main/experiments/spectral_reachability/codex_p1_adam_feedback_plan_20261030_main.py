"""Plan a refreshed final packet for every fixed Adam physical trajectory."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_rgb_gap_20261030_main as gap
from . import codex_p1_same_first5_feedback_compare_20261030_main as traces
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback

base = gap.base
NAME = "codex_p1_adam_feedback_plan_20261030_main"


def inputs():
    data = {name: gap.prior.gap.read(base.ROOT / path, digest)[0]
            for name, (path, digest) in gap.joined.SOURCES.items()}
    gap.joined.build(**data)
    bundle, _ = gap.prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    report, _ = gap.prior.gap.read(gap.PHYSICAL / "run_report.json", gap.REPORT_SHA)
    if not all(len(x) == 24 for x in (bundle["rows"], report["shards"], data["saved"]["rows"])):
        raise ValueError("fixed roster")
    entries = []
    for i, (row, p, s, plan, shard) in enumerate(zip(
            bundle["rows"], data["prepare"]["roster"], data["saved"]["rows"], data["plan"]["rows"],
            report["shards"], strict=True)):
        if not (row["index"] == p["index"] == s["index"] == plan["index"] == shard["index"] == i):
            raise ValueError("identity")
        raw = np.asarray(plan["selected_actions"], dtype="<f4")
        if raw.shape != (2, 5, 2) or hashlib.sha256(raw.tobytes()).hexdigest() != p["raw10_sha256"]:
            raise ValueError("plan action identity")
        folder = gap.PHYSICAL / f"row_{i:02d}"
        ts = traces.transitions(folder, shard["report_sha256"])
        if len(ts) < 5 or any(t["step"] != j + 1 or t["actual_bits"] != raw[0, j].tobytes().hex()
                              for j, t in enumerate(ts[:5])):
            raise ValueError("executed first packet")
        report_row, _ = gap.prior.gap.read(folder / "report.json", shard["report_sha256"])
        frames = [f for f in report_row["row"]["trajectory"] if f["phase"] == "probe" and f["step"] == 5
                  and f["condition"] == s["condition"]
                  and f["overlay_index"] == (1 if s["condition"] == "on" else 0)]
        if len(frames) != 1:
            raise ValueError("midpoint frame")
        frame = frames[0]
        png = folder / frame["png"]
        if Path(frame["png"]).name != frame["png"] or base.sha(png) != frame["file_sha256"] or frame["state"] != ts[4]["state"]:
            raise ValueError("midpoint binding")
        with Image.open(png) as image:
            rgb = np.asarray(image).copy()
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8 or hashlib.sha256(rgb.tobytes()).hexdigest() != frame["pixel_sha256"]:
            raise ValueError("midpoint pixels")
        entries.append((row, s, raw, rgb, frame))
    return entries


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    entries = inputs()
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        tx = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, s, raw, rgb, frame in entries:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                with torch.no_grad():
                    encode = lambda x: model.encode({"pixels": p0._image_tensor(x, transform=tx, device=device)})["emb"][0, 0]
                    cp5, midpoint = encode(cp5_rgb), encode(rgb)
                    states, history, goal, mean, std = feedback.second_context(row, cp5, midpoint, raw[0])
                action = torch.nn.Parameter(torch.tensor(raw[1][None], dtype=torch.float32, device=device))
                optimizer = torch.optim.Adam([action], lr=0.05)
                def score():
                    token = ((action - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                    return (pred - goal).square().mean()
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
                torch.cuda.synchronize(device)
                # Fixed rule: keep the original packet when its refreshed-model score is lower.
                chosen = action.detach().cpu().numpy()[0].astype("<f4") if final < baseline else raw[1].copy()
                rows.append({"index": row["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "first_packet_hex": raw[0].tobytes().hex(), "second_packet_hex": chosen.tobytes().hex(),
                             "midpoint_pixel_sha256": frame["pixel_sha256"], "baseline_score": baseline,
                             "optimized_score": final, "chosen": "optimized" if final < baseline else "original",
                             "optimizer_seconds": time.monotonic() - start})
        if before != base.model_hash(model):
            raise ValueError("model mutated")
    result = {"schema": NAME, "status": "complete", "rows": rows, "host": host,
              "model_before": before, "model_after": before, "code_sha256": base.sha(Path(__file__).resolve()),
              "source_hashes": gap.joined.SOURCES, "physical_report_sha256": gap.REPORT_SHA,
              "per_row_calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20},
              "P1_admission": False, "independent_confirmation": False,
              "limitations": "known roots; post-outcome design; offline real-midpoint replay, not online control; "
                             "extra compute vs open; explicit path counts not full budget certification"}
    print(json.dumps({"rows": len(rows), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Optimize second Adam packet from authenticated real RGB midpoint; fixed 24 rows.", flush=True)
    run(out)
