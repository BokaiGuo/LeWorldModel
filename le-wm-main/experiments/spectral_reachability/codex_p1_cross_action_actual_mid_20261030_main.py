"""Evaluate both executed second packets at the same real midpoint on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import codex_p1_predicted_midpoint_replay_20261030_main as previous

NAME = "codex_p1_cross_action_actual_mid_20261030_main"
base, compare, pred = previous.base, previous.compare, previous.pred
pair = compare.pair


def frame(folder, report, condition, step):
    matches = [f for f in report["row"]["trajectory"] if f["phase"] == "probe"
               and f["step"] == step and f["condition"] == condition
               and f["overlay_index"] == (1 if condition == "on" else 0)]
    pair.need(len(matches) == 1, "frame identity")
    f = matches[0]
    path = folder/"physical"/f["png"]
    pair.need(Path(f["png"]).name == f["png"] and base.sha(path) == f["file_sha256"], "frame file hash")
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    pair.need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
              and hashlib.sha256(pixels.tobytes()).hexdigest() == f["pixel_sha256"], "frame pixels")
    return pixels, f["pixel_sha256"]


def action(folder, transitions, physical_report):
    stage_path = folder/"stage_2/report.json"
    events = pair.journal.events(folder/"physical/journal.jsonl", physical_report["journal_chain_tail"])
    plans = [e for e in events if e["kind"] == "phase_plan_returned" and e["phase"] == 2]
    pair.need(len(plans) == 1 and base.sha(stage_path) == plans[0]["stage_report_sha256"],
              "phase two report binding")
    report = json.loads(stage_path.read_bytes())
    raw = bytes.fromhex(report["result"]["raw_horizon_f32_hex"])
    pair.need(len(raw) == 80 and len(transitions) == 10
              and hashlib.sha256(raw).hexdigest() == plans[0]["plan_sha256"], "full second plan")
    packet = np.frombuffer(raw[:40], dtype="<f4").copy().reshape(5, 2)
    for i in range(5):
        pair.need(transitions[i+5]["step"] == i+6
                  and transitions[i+5]["actual_bits"] == packet[i].tobytes().hex(), "executed packet")
    return packet


def run(out):
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0
    p_saved, r_saved = pair.read(compare.PRED_SAVED, compare.PRED_SHA), pair.read(pair.FB_SAVED, pair.FB_SHA)
    p_report = pair.read(compare.PRED_DIR/"run_report.json", compare.PRED_REPORT)
    r_report = pair.read(pair.replay.FB_DIR/"run_report.json", pair.replay.FB_REPORT_SHA)
    bundle = pair.read(base.BUNDLE, base.BUNDLE_SHA)
    need = pair.need
    need(all(len(x["rows"]) == 24 for x in (p_saved, r_saved, bundle)), "fixed roster")
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "checkpoint identity")
        tx = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for i, (p, r, row) in enumerate(zip(p_saved["rows"], r_saved["rows"], bundle["rows"], strict=True)):
                need(all(p[k] == r[k] for k in pair.IDS) and p["index"] == row["index"] == i
                     and p["status"] == r["status"] == "probe_completed", "identity/endpoint")
                folders = (compare.PRED_DIR/f"row_{i:02d}", pair.replay.FB_DIR/f"row_{i:02d}")
                shards = (p_report["shards"][i], r_report["shards"][i])
                transitions, reports, packets, terminal = [], [], [], []
                for folder, shard in zip(folders, shards, strict=True):
                    need(shard["index"] == i, "shard index")
                    report = pair.read(folder/"physical/report.json", shard["report_sha256"])
                    ts = pair.transitions(folder/"physical", shard["report_sha256"])
                    reports.append(report)
                    transitions.append(ts)
                    packets.append(action(folder, ts, report))
                    terminal.append(frame(folder, report, p["condition"], 10))
                for j in range(5):
                    need(all(transitions[0][j][k] == transitions[1][j][k] for k in
                             ("step", "actual_bits", "state", "snapshot", "terminated", "truncated")), "prefix mismatch")
                first_packet = np.frombuffer(b"".join(bytes.fromhex(t["actual_bits"])
                                                       for t in transitions[0][:5]), dtype="<f4").copy().reshape(5, 2)
                mid_rgb, mid_sha = frame(folders[1], reports[1], p["condition"], 5)
                mid_other, mid_other_sha = frame(folders[0], reports[0], p["condition"], 5)
                need(mid_sha == mid_other_sha and np.array_equal(mid_rgb, mid_other), "midpoint RGB mismatch")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as img:
                    cp5 = np.asarray(img).copy()
                encode = lambda img: model.encode({"pixels": p0._image_tensor(img, transform=tx, device=device)})["emb"][0, 0]
                z5, mid = encode(cp5), encode(mid_rgb)
                states, history, goal, mean, std = pred.bridge.second_context(row, z5, mid, first_packet)
                candidates = []
                for packet, (rgb, pixel_sha), item in zip(packets, terminal, (p, r), strict=True):
                    token = ((torch.as_tensor(packet, device=device)-mean)/std).reshape(1, 1, 10)
                    predicted = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[0, -1]
                    actual = encode(rgb)
                    need(bool(torch.isfinite(predicted).all()) and bool(torch.isfinite(actual).all()), "nonfinite")
                    mse = lambda a, b: float((a-b).square().mean().item())
                    candidates.append({"packet_sha256": hashlib.sha256(packet.tobytes()).hexdigest(),
                                       "terminal_pixel_sha256": pixel_sha,
                                       "predicted_goal_mse": mse(predicted, goal[0]),
                                       "actual_goal_mse": mse(actual, goal[0]),
                                       "prediction_error_mse": mse(predicted, actual),
                                       "predicted_latent": predicted.cpu().tolist(),
                                       "actual_latent": actual.cpu().tolist(),
                                       "physical_metrics": item["endpoint_metrics"]})
                rows.append({"identity": {k: p[k] for k in pair.IDS}, "midpoint_pixel_sha256": mid_sha,
                             "predicted_context_action": candidates[0], "real_context_action": candidates[1]})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before, "model changed")
    result = {"schema": NAME, "status": "complete", "rows": rows, "model_before": before,
              "model_after": before, "host": host, "code_sha256": base.sha(Path(__file__).resolve()),
              "sources": {str(compare.PRED_SAVED): compare.PRED_SHA, str(pair.FB_SAVED): pair.FB_SHA,
                          str(compare.PRED_DIR/"run_report.json"): compare.PRED_REPORT,
                          str(pair.replay.FB_DIR/"run_report.json"): pair.replay.FB_REPORT_SHA,
                          str(base.BUNDLE): base.BUNDLE_SHA},
              "explicit_calls": {"encoder": 96, "projector": 96, "action_encoder": 48, "predict": 48},
              "limitations": "known roots and repeated actions; post-outcome diagnostic, same model ABI; "
                             "call counts are explicit code-path, not independent hooks",
              "independent_confirmation": False, "P1_admission": False}
    print(json.dumps({"rows": len(rows), "sha256": base.write(out, result)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    out = Path(ap.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT/"results" or not out.name.startswith(NAME+"_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Replay both fixed executed actions from the same actual midpoint.", flush=True)
    run(out)
