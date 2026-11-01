"""Diagnose two fixed cp10 suffixes under the same real RGB context."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_hybrid_cp10_feedback_saved_20261031 as feedback_saved
from . import p1_hybrid_cp5_physical_saved_20261031 as open_saved
from . import p1_hybrid_cp10_feedback_cuda_20261031 as feedback_plan

ROOT = feedback_plan.ROOT
base, need, sha = feedback_plan.base, feedback_plan.need, feedback_plan.sha
NAME = "p1_cp10_paired_prediction_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_CP10_PAIRED_PREDICTION_PROTOCOL_20261031.md"
FEEDBACK_DIR = ROOT / "results/p1_hybrid_cp10_feedback_physical_20261031_RUN01"
FEEDBACK_REPORT_SHA = "e0f214e8feb5644003e58d673b72f67e08dcf7d6d01b01d213a2a75d75e96fe5"
FEEDBACK_PREPARE = ROOT / "results/p1_hybrid_cp10_feedback_physical_20261031_prepare_RUN02.json"
FEEDBACK_PREPARE_SHA = "50536c52d9d2830356d4f9a11596703b038d6507dbe36cb56fa77780abbfd2c7"
PAIRED = ROOT / "results/p1_hybrid_cp10_feedback_paired_20261031_RUN01.json"
PAIRED_SHA = "ddcf2c13bfa57c5fa19b21f8099efe9467faf6e70ac795fa7200ee5748f53acb"


def load_image(path, pixel_sha):
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
         and hashlib.sha256(pixels.tobytes()).hexdigest() == pixel_sha, "physical pixel identity")
    return pixels


def inputs():
    source = feedback_plan.source
    verified_open = open_saved.consume(source.DIRECTORY, source.REPORT_SHA, source.PREPARE, source.PREPARE_SHA)
    verified_feedback = feedback_saved.consume(FEEDBACK_DIR, FEEDBACK_REPORT_SHA,
                                                FEEDBACK_PREPARE, FEEDBACK_PREPARE_SHA)
    need(sha(PAIRED) == PAIRED_SHA and len(verified_open["rows"]) == len(verified_feedback["rows"]) == 2,
         "saved sources")
    paired = source.physical.h.read(PAIRED)
    plan = source.physical.h.read(feedback_plan.PLAN if hasattr(feedback_plan, "PLAN")
                                  else ROOT / "results/p1_hybrid_cp10_feedback_cuda_20261031_RUN01.json")
    bundle = source.physical.h.read(source.physical.BUNDLE)
    old_p = source.physical.h.read(source.PREPARE)
    new_p = source.physical.h.read(FEEDBACK_PREPARE)
    old_run = source.physical.h.read(source.DIRECTORY / "run_report.json")
    new_run = source.physical.h.read(FEEDBACK_DIR / "run_report.json")
    sources = dict(new_p["bindings"])
    for path in (PAIRED, FEEDBACK_PREPARE, FEEDBACK_DIR / "run_report.json", source.PREPARE,
                 source.DIRECTORY / "run_report.json", PROTOCOL, Path(__file__).resolve()):
        sources[str(path)] = sha(path)
    items = []
    for anchor, blind, old_action, new_action, candidate, old_shard, new_shard, physical in zip(
            old_p["anchors"], bundle["rows"], old_p["items"], new_p["items"], plan["rows"],
            old_run["shards"], new_run["shards"], paired["rows"], strict=True):
        index = anchor["index"]
        need(index == old_action["index"] == new_action["index"] == candidate["index"] ==
             old_shard["index"] == new_shard["index"] and physical["root_slot"] == anchor["root_slot"],
             "root and task identity")
        old_raw = bytes.fromhex(old_action["raw10_float32_hex"])
        new_raw = bytes.fromhex(new_action["raw10_float32_hex"])
        need(len(old_raw) == len(new_raw) == 80 and old_raw[:40] == new_raw[:40], "first action packet")
        frames = {}
        for label, directory, shard in (("open", source.DIRECTORY, old_shard),
                                         ("feedback", FEEDBACK_DIR, new_shard)):
            folder = directory / f"row_{index:02d}"
            report_path = folder / "report.json"
            report = source.physical.h.read(report_path)
            journal = folder / "journal.jsonl"
            need(sha(report_path) == shard["sha256"] and sha(journal) == report["journal_sha256"]
                 and report["row"]["status"] == "probe_completed", "complete physical row")
            events = open_saved.saved.events(journal, report["journal_chain_tail"])
            matches = [e["receipt"] for e in events if e["kind"] == "fresh_observation"
                       and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] in (5, 10)
                       and e["receipt"]["overlay_index"] == 0]
            need([r["step"] for r in matches] == [5, 10], "midpoint and endpoint frames")
            frames[label] = []
            for rec in matches:
                path = folder / rec["png"]
                need(path.parent == folder and not path.is_symlink() and sha(path) == rec["file_sha256"],
                     "image file SHA")
                frames[label].append((path, rec))
                sources[str(path)] = rec["file_sha256"]
            sources[str(report_path)], sources[str(journal)] = shard["sha256"], report["journal_sha256"]
        need(all(frames["open"][0][1][k] == frames["feedback"][0][1][k]
                 for k in ("pixel_sha256", "state", "snapshot")), "same cp10")
        items.append({"anchor": anchor, "blind": blind["blind"], "candidate": candidate,
                      "physical": physical, "frames": frames,
                      "open_raw": np.frombuffer(old_raw, "<f4").copy().reshape(2, 5, 2),
                      "feedback_raw": np.frombuffer(new_raw, "<f4").copy().reshape(2, 5, 2)})
    need(all(sha(p) == digest for p, digest in sources.items()), "source drift")
    return items, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    items, sources = inputs()
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen model")
        transform, rows = p0.image_transform(), []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for item in items:
                blind = item["blind"]
                with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as im:
                    cp5_rgb = np.asarray(im).copy()
                mid_path, mid_frame = item["frames"]["open"][0]
                midpoint_rgb = load_image(mid_path, mid_frame["pixel_sha256"])
                def encode(pixels):
                    return model.encode({"pixels": p0._image_tensor(pixels, transform=transform,
                                                                      device=device)})["emb"][0, 0]
                cp5, cp10 = encode(cp5_rgb), encode(midpoint_rgb)
                states, history, goal, mean, std = base.bridge.cp5.context(blind, cp5, device=device)
                states = torch.cat((states, cp10[None, None]), dim=1)[:, -3:]
                first = torch.tensor(item["open_raw"][0], dtype=torch.float32, device=device)
                token = ((first - mean) / std).reshape(1, 1, 10)
                history = torch.cat((history, token), dim=1)[:, -2:]
                predictions = {}
                for label in ("open", "feedback"):
                    raw = item[label + "_raw"][1]
                    norm = ((torch.tensor(raw, device=device) - mean) / std).reshape(1, 1, 10)
                    pred = model.predict(states, model.action_encoder(torch.cat((history, norm), 1)))[0, -1]
                    path, rec = item["frames"][label][1]
                    actual = encode(load_image(path, rec["pixel_sha256"]))
                    predictions[label] = {
                        "predicted_goal_mse": float((pred - goal[0]).square().mean()),
                        "real_goal_mse": float((actual - goal[0]).square().mean()),
                        "prediction_real_mse": float((pred - actual).square().mean()),
                        "prediction": pred.cpu().tolist(), "actual": actual.cpu().tolist(),
                        "terminal_pixel_sha256": rec["pixel_sha256"],
                        "action_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                        "physical": item["physical"][label]}
                winner = item["candidate"]["selected_index"]
                need(abs(predictions["feedback"]["predicted_goal_mse"] -
                         item["candidate"]["candidate_scores"][winner]) <= 1e-5, "planner score replay")
                rows.append({"root_slot": item["anchor"]["root_slot"], "goal": goal[0].cpu().tolist(),
                             "midpoint_pixel_sha256": mid_frame["pixel_sha256"], "arms": predictions})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(sha(p) == d for p, d in sources.items()),
             "model/source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "host": host,
                  "model_before": before, "model_after": before, "sources": sources,
                  "calls": {"encoder": 8, "action_encoder": 4, "predict": 4},
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Known roots and post-outcome diagnosis; no new physical execution."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
