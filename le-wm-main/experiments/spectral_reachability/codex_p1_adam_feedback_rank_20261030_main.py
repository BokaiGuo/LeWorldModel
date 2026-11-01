"""Compare both executed suffixes at the same authenticated real RGB midpoint."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_feedback_compare_20261030_main as paired

base = paired.base
plan = paired.plan
NAME = "codex_p1_adam_feedback_rank_20261030_main"
PLAN = base.ROOT / "results/codex_p1_adam_feedback_plan_20261030_main_RUN01.json"
PLAN_SHA = "3fe7a35c3b42600a29d74423cdd80cf75a85499a54b31e276dd03261b4ae50c5"


def frame(folder, shard, condition, step, terminal_state):
    report, _ = plan.gap.prior.gap.read(folder / "report.json", shard["report_sha256"])
    frames = [f for f in report["row"]["trajectory"] if f["phase"] == "probe" and f["step"] == step
              and f["condition"] == condition and f["overlay_index"] == (1 if condition == "on" else 0)]
    if len(frames) != 1:
        raise ValueError("terminal frame count")
    f = frames[0]
    path = folder / f["png"]
    if Path(f["png"]).name != f["png"] or base.sha(path) != f["file_sha256"] or f["state"] != terminal_state:
        raise ValueError("terminal file/state")
    with Image.open(path) as image:
        rgb = np.asarray(image).copy()
    if rgb.shape != (224, 224, 3) or hashlib.sha256(rgb.tobytes()).hexdigest() != f["pixel_sha256"]:
        raise ValueError("terminal pixels")
    return rgb, f["pixel_sha256"]


def inputs():
    pairs = paired.build()
    fixed = plan.inputs()
    optimized, _ = plan.gap.prior.gap.read(PLAN, PLAN_SHA)
    old_report, _ = plan.gap.prior.gap.read(plan.gap.PHYSICAL / "run_report.json", plan.gap.REPORT_SHA)
    new_report, _ = plan.gap.prior.gap.read(paired.DIR / "run_report.json", paired.REPORT_SHA)
    old_saved, _ = plan.gap.prior.gap.read(base.ROOT / plan.gap.joined.SOURCES["saved"][0], plan.gap.joined.SOURCES["saved"][1])
    new_saved, _ = plan.gap.prior.gap.read(paired.SAVED, paired.SAVED_SHA)
    if not all(len(x) == 24 for x in (pairs["rows"], fixed, optimized["rows"], old_report["shards"],
                                      new_report["shards"], old_saved["rows"], new_saved["rows"])):
        raise ValueError("roster")
    entries = []
    for i, (pair, entry, opt, old, new) in enumerate(zip(
            pairs["rows"], fixed, optimized["rows"], old_saved["rows"], new_saved["rows"], strict=True)):
        if pair["index"] != i or opt["index"] != i:
            raise ValueError("identity")
        if pair["feedback_minus_open"] is None:
            continue
        row, physical, raw, midpoint, midframe = entry
        packets = [raw[1], np.frombuffer(bytes.fromhex(opt["second_packet_hex"]), dtype="<f4").copy().reshape(5, 2)]
        folders = [plan.gap.PHYSICAL / f"row_{i:02d}", paired.DIR / f"row_{i:02d}"]
        shards = [old_report["shards"][i], new_report["shards"][i]]
        outcomes = [old, new]
        terminal = []
        for folder, shard, packet, outcome in zip(folders, shards, packets, outcomes, strict=True):
            ts = paired.trace.transitions(folder, shard["report_sha256"])
            if len(ts) != 10 or any(ts[j+5]["actual_bits"] != packet[j].tobytes().hex() for j in range(5)):
                raise ValueError("executed second packet mismatch")
            terminal.append(frame(folder, shard, old["condition"], 10, outcome["completed_endpoint"]))
        entries.append((row, physical, raw[0], midpoint, midframe, opt, packets, terminal, pair))
    if len(entries) != 19:
        raise ValueError("uncensored denominator")
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
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for row, physical, first, midpoint, midframe, opt, packets, terminal, pair in entries:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                encode = lambda x: model.encode({"pixels": p0._image_tensor(x, transform=tx, device=device)})["emb"][0, 0]
                cp5, mid = encode(cp5_rgb), encode(midpoint)
                states, history, goal, mean, std = plan.feedback.second_context(row, cp5, mid, first)
                candidates = []
                for packet, (rgb, pixel_sha) in zip(packets, terminal, strict=True):
                    token = ((torch.as_tensor(packet, device=device)-mean)/std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[0, -1]
                    actual = encode(rgb)
                    mse = lambda a, b: float((a-b).square().mean().item())
                    candidates.append({"predicted_goal_mse": mse(pred, goal[0]), "actual_goal_mse": mse(actual, goal[0]),
                                       "prediction_error_mse": mse(pred, actual), "pixel_sha256": pixel_sha,
                                       "predicted_latent": pred.cpu().tolist(), "actual_latent": actual.cpu().tolist()})
                if abs(candidates[0]["predicted_goal_mse"] - opt["baseline_score"]) > 1e-5:
                    raise ValueError("baseline score mismatch")
                expected = opt["optimized_score"] if opt["chosen"] == "optimized" else opt["baseline_score"]
                if abs(candidates[1]["predicted_goal_mse"] - expected) > 1e-5:
                    raise ValueError("selected score mismatch")
                rows.append({"index": row["index"], "logical_pair": physical["logical_pair"], "role": physical["role"],
                             "root_slot": physical["root_slot"], "condition": physical["condition"],
                             "midpoint_pixel_sha256": midframe["pixel_sha256"], "open": candidates[0],
                             "feedback": candidates[1], "physical_delta": pair["feedback_minus_open"]})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model):
            raise ValueError("model changed")
    result = {"schema": NAME, "status": "complete", "rows": rows, "fixed_roster": 24, "censored": 5,
              "host": host, "model_before": before, "model_after": before,
              "code_sha256": base.sha(Path(__file__).resolve()), "plan_sha256": PLAN_SHA,
              "source_report_sha256": plan.gap.REPORT_SHA, "feedback_report_sha256": paired.REPORT_SHA,
              "P1_admission": False, "independent_confirmation": False,
              "limitations": "post-outcome analysis on two known roots, repeated trajectories; "
                             "excludes censored successes, no causal generalization"}
    print(json.dumps({"rows": len(rows), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Replay both fixed executed packets from the same real RGB midpoint on host CUDA.", flush=True)
    run(out)
