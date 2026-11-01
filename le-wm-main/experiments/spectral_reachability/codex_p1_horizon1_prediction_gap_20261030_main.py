"""Replay the two executed horizon variants from the same observed midpoint."""
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

from . import codex_p1_horizon1_compare_20261030_main as comp
from . import codex_p1_cross_action_actual_mid_20261030_main as cross

NAME = "codex_p1_horizon1_prediction_gap_20261030_main"
base, pair, h1 = cross.base, cross.pair, comp.h1
MATCH = h1.ROOT / "results/codex_p1_horizon1_compare_20261030_main_RUN01.json"
MATCH_SHA = "0368e933d4de4d68c776d3abea8cd88f0912a4569df139b057379d3807e7a971"


def frame(folder, report, condition, step):
    frames = [f for f in report["row"]["trajectory"] if f["phase"] == "probe"
              and f["step"] == step and f["condition"] == condition
              and f["overlay_index"] == (1 if condition == "on" else 0)]
    pair.need(len(frames) == 1 and Path(frames[0]["png"]).name == frames[0]["png"], "frame identity")
    f = frames[0]
    path = folder / f["png"]
    pair.need(base.sha(path) == f["file_sha256"], "frame file hash")
    with Image.open(path) as image:
        rgb = np.asarray(image).copy()
    pair.need(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
              and hashlib.sha256(rgb.tobytes()).hexdigest() == f["pixel_sha256"], "RGB pixels")
    return rgb, f["pixel_sha256"]


def run(out):
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    match = pair.read(MATCH, MATCH_SHA)
    one = pair.read(comp.SAVED, comp.SAVED_SHA)
    two = pair.read(comp.pair.FB_SAVED, comp.pair.FB_SHA)
    report_one = pair.read(comp.DIR / "run_report.json", comp.REPORT)
    report_two = pair.read(comp.pair.replay.FB_DIR / "run_report.json", comp.pair.replay.FB_REPORT_SHA)
    bundle = pair.read(base.BUNDLE, base.BUNDLE_SHA)
    plan = pair.read(h1.PLAN, h1.PLAN_SHA)
    pair.need(all(len(x["rows"]) == 24 for x in (match, one, two, bundle, plan)), "fixed roster")
    results = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        pair.need(before == base.MODEL_HASH, "model identity")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for i, (m, a, b, row, p) in enumerate(zip(
                    match["rows"], one["rows"], two["rows"], bundle["rows"], plan["rows"], strict=True)):
                pair.need(a["index"] == b["index"] == row["index"] == p["index"] == i
                          and all(a[k] == b[k] == m["identity"][k] for k in comp.pair.IDS)
                          and a["status"] == b["status"] == "probe_completed", "identity and completion")
                folders = (comp.DIR / f"row_{i:02d}",
                           comp.pair.replay.FB_DIR / f"row_{i:02d}" / "physical")
                shards = (report_one["shards"][i], report_two["shards"][i])
                reports = [pair.read(f / "report.json", s["report_sha256"])
                           for f, s in zip(folders, shards, strict=True)]
                traces = [comp.pair.transitions(f, s["report_sha256"])
                          for f, s in zip(folders, shards, strict=True)]
                pair.need(all(len(t) == 10 for t in traces), "ten actions")
                for j in range(5):
                    pair.need(all(traces[0][j][k] == traces[1][j][k]
                                  for k in ("step", "actual_bits", "state", "snapshot", "terminated", "truncated")),
                              "midpoint dynamics")
                first = np.frombuffer(b"".join(bytes.fromhex(t["actual_bits"]) for t in traces[0][:5]),
                                      dtype="<f4").copy().reshape(5, 2)
                pair.need(first.tobytes().hex() == p["first_packet_hex"], "first action binding")
                mids = [frame(f, r, a["condition"], 5) for f, r in zip(folders, reports, strict=True)]
                pair.need(mids[0][1] == mids[1][1] == p["midpoint_pixel_sha256"], "midpoint RGB")
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5 = np.asarray(image).copy()
                encode = lambda image: model.encode(
                    {"pixels": p0._image_tensor(image, transform=transform, device=device)})["emb"][0, 0]
                z5, zmid = encode(cp5), encode(mids[0][0])
                states, history, goal, mean, std = cross.pred.bridge.second_context(row, z5, zmid, first)
                candidates = []
                for j, (folder, report, trace, outcome) in enumerate(zip(
                        folders, reports, traces, (a, b), strict=True)):
                    packet = np.frombuffer(b"".join(bytes.fromhex(t["actual_bits"]) for t in trace[5:]),
                                           dtype="<f4").copy().reshape(5, 2)
                    if j == 0:
                        pair.need(packet.tobytes().hex() == p["second_packet_hex"], "single packet binding")
                    else:
                        bound = cross.action(folder.parent, trace, report)
                        pair.need(np.array_equal(packet, bound), "two packet binding")
                    token = ((torch.as_tensor(packet, device=device) - mean) / std).reshape(1, 1, 10)
                    predicted = model.predict(states, model.action_encoder(torch.cat((history, token), 1)))[0, -1]
                    terminal_rgb, terminal_sha = frame(folder, report, a["condition"], 10)
                    actual = encode(terminal_rgb)
                    pair.need(bool(torch.isfinite(predicted).all()) and bool(torch.isfinite(actual).all()),
                              "finite encoding")
                    mse = lambda x, y: float((x-y).square().mean().item())
                    candidates.append({"packet_sha256": hashlib.sha256(packet.tobytes()).hexdigest(),
                                       "terminal_pixel_sha256": terminal_sha,
                                       "predicted_goal_mse": mse(predicted, goal[0]),
                                       "actual_goal_mse": mse(actual, goal[0]),
                                       "prediction_error_mse": mse(predicted, actual),
                                       "predicted_latent": predicted.cpu().tolist(),
                                       "actual_latent": actual.cpu().tolist(),
                                       "physical_metrics": outcome["endpoint_metrics"]})
                results.append({"identity": m["identity"], "midpoint_pixel_sha256": mids[0][1],
                                "one_packet": candidates[0], "two_packet": candidates[1]})
        torch.cuda.synchronize(device)
        pair.need(base.model_hash(model) == before, "model mutation")
    result = {"schema": NAME, "status": "complete", "rows": results, "host": host,
              "model_before": before, "model_after": before, "match_sha256": MATCH_SHA,
              "code_sha256": base.sha(Path(__file__).resolve()),
              "explicit_calls": {"encoder": 96, "projector": 96, "action_encoder": 48, "predict": 48},
              "limitations": "post-outcome known roots and repeated trajectories; explicit code-path "
                             "counts, not hook-audited; same model ABI; no P1 or independent confirmation",
              "P1_admission": False}
    print(json.dumps({"rows": len(results), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    pair.need(output.is_absolute() and output.parent == base.ROOT / "results"
              and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
              "exclusive output")
    print("Compare predictions and real RGB encodings for both already executed packets.", flush=True)
    run(output)
