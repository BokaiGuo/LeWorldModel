"""Replay executed suffixes and compare actual midpoint/terminal RGB on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_response_prefix_cp5_physical_20261031 as physical
from . import codex_p1_multistart_cp5_20261030_main as common
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback
from . import codex_p1_h10_paired_table_20261030_main as metric

ROOT, need, sha = physical.ROOT, physical.need, physical.sha
NAME = "p1_response_cp5_same_action_20261031"
PREP = ROOT / "results/p1_response_prefix_cp5_physical_20261031_prepare_RUN01.json"
PREP_SHA = "0aa3b133b22e58e1673fb546ef8a4a7679a8c17c5b652cae9fbabc6c012c2102"
RUN = ROOT / "results/p1_response_prefix_cp5_physical_20261031_RUN01"
REPORT_SHA = "0b64f7f65faae3bb7c13aa4b62bb5a53713480c05146f8b61597b52fe93b4aaa"
RECEIPT = ROOT / "results/p1_response_prefix_cp5_physical_20261031_consume_RUN01.json"
RECEIPT_SHA = "7f7c67cef27a790e9d64eacde6d707c92d0cf0c027bd9901c6c0825794399f75"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_CP5_SAME_ACTION_PROTOCOL_20261031.md"


def collect():
    read = physical.bundle.physical.prior.native.read
    paths = {PREP: PREP_SHA, RUN / "run_report.json": REPORT_SHA, RECEIPT: RECEIPT_SHA,
             physical.BUNDLE: physical.BUNDLE_SHA, physical.PLAN: physical.PLAN_SHA}
    need(all(sha(path) == digest for path, digest in paths.items()), "sealed inputs")
    prepared, run, receipt, bundle, plan = [read(p, cap=64 * 1024**2) for p in paths]
    need(receipt["status"] == "saved_checks_passed" and len(receipt["rows"]) == 14 and
         run["status"] == "recorded_not_consumed" and len(run["shards"]) == 14 and
         len(prepared["roster"]) == len(prepared["anchors"]) == 14 and len(bundle["rows"]) == 16 and
         len(plan["rows"]) == 16, "fixed sealed roster")
    sources = {str(path): digest for path, digest in paths.items()}
    for path in (Path(__file__).resolve(), PROTOCOL, Path(feedback.__file__).resolve(),
                 Path(metric.__file__).resolve()):
        sources[str(path)] = sha(path)
    entries, excluded = [], [{"index": i, "status": "censored_before_cp5"}
                             for i in prepared["censored_before_cp5"]]
    for anchor, item, shard, verified in zip(prepared["anchors"], prepared["roster"],
                                              run["shards"], receipt["rows"], strict=True):
        i = item["index"]
        need(i == anchor["index"] == shard["index"] == verified["index"] and
             shard["status"] == "recorded_not_consumed", "physical row identity")
        blind, proposed = bundle["rows"][i]["blind"], plan["rows"][i]
        need(blind is not None and proposed["status"] == "planned" and
             blind["cp5_pixel_sha256"] == proposed["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"]
             and blind["prefix_raw_f32_hex"] == item["prefix5_float32_hex"], "planner input binding")
        raw = np.asarray(proposed["actions"], dtype="<f4")
        need(raw.shape == (2, 5, 2) and np.isfinite(raw).all() and
             hashlib.sha256(raw.tobytes()).hexdigest() == item["raw10_sha256"] == proposed["actions_sha256"]
             and raw.tobytes().hex() == item["raw10_float32_hex"], "physical action binding")
        path = RUN / f"row_{i:02d}" / "report.json"
        need(sha(path) == shard["sha256"], "row report")
        report = read(path, cap=64 * 1024**2)
        row = report["row"]
        need(row["status"] == verified["status"], "status binding")
        sources[str(path)] = shard["sha256"]
        if row["status"] != "probe_completed":
            excluded.append({"index": i, "status": row["status"], "returned_suffix": verified["returned_suffix"]})
            continue
        need(len(row["responses"]) == 10 and verified["completed_endpoint"] == row["responses"][-1]["state"],
             "terminal physical binding")
        images = []
        for step in (5, 10):
            frames = [x for x in row["trajectory"] if x["phase"] == "probe" and x["step"] == step]
            need(len(frames) == 1 and frames[0]["state"] == row["responses"][step - 1]["state"] and
                 frames[0]["condition"] == item["condition"], "same-step image state")
            frame = frames[0]
            image_path = path.parent / frame["png"]
            need(image_path.parent == path.parent and sha(image_path) == frame["file_sha256"], "image file")
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            need(pixels.shape == (224, 224, 3) and hashlib.sha256(pixels.tobytes()).hexdigest() ==
                 frame["pixel_sha256"], "image pixels")
            sources[str(image_path)] = frame["file_sha256"]
            images.append(pixels)
        entries.append({"index": i, "logical_pair": bundle["rows"][i]["logical_pair"],
                        "condition": item["condition"], "arm": item["arm"], "role": item["role"],
                        "root_slot": item["root_slot"], "blind": blind, "raw": raw, "mid": images[0],
                        "terminal": images[1], "score": proposed["candidate_scores"][proposed["selected_index"]],
                        "physical_start": metric.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"]),
                        "physical_end": metric.physical_metrics(verified["completed_endpoint"], anchor["goal"]),
                        "action_sha256": item["raw10_sha256"]})
    need(len(entries) + len(excluded) == 16 and len({e["index"] for e in entries + excluded}) == 16,
         "complete denominator")
    return entries, excluded, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = common.base
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results" and output.name.startswith(NAME + "_")
         and not output.exists() and not output.is_symlink(), "exclusive output")
    entries, excluded, sources = collect()
    for path, digest in base.bindings().items():
        need(path not in sources or sources[path] == digest, "source collision")
        sources[path] = digest
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen checkpoint")
        transform = p0.image_transform()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                row = entry["blind"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_pixels = np.asarray(image.convert("RGB")).copy()
                def encode(pixels):
                    return model.encode({"pixels": p0._image_tensor(pixels, transform=transform,
                                                                    device=device)})["emb"][0, 0]
                cp, mid, actual = encode(cp_pixels), encode(entry["mid"]), encode(entry["terminal"])
                actions = torch.as_tensor(entry["raw"][None], dtype=torch.float32, device=device)
                predicted, goal = common.fixed.replay(model, row, cp, actions)
                score = float((predicted - goal).square().mean())
                need(np.isfinite(score) and abs(score - entry["score"]) <= 1e-5, "replay score mismatch")
                states, history, goal, mean, std = feedback.second_context(row, cp, mid, entry["raw"][0])
                token = ((torch.as_tensor(entry["raw"][1], device=device) - mean) / std).reshape(1, 1, 10)
                refreshed = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[0, -1]
                predicted = predicted.reshape(-1)
                actual_goal = float((actual - goal[0]).square().mean())
                value = {k: entry[k] for k in ("index", "logical_pair", "condition", "arm", "role", "root_slot",
                                                "physical_start", "physical_end", "action_sha256")}
                value.update(predicted_goal_mse=score, actual_goal_mse=actual_goal,
                             refreshed_goal_mse=float((refreshed - goal[0]).square().mean()),
                             open_rgb_mse=float((predicted - actual).square().mean()),
                             refreshed_rgb_mse=float((refreshed - actual).square().mean()),
                             predicted_latent=predicted.cpu().tolist(), refreshed_latent=refreshed.cpu().tolist(),
                             actual_latent=actual.cpu().tolist(), goal_latent=goal[0].cpu().tolist())
                need(all(np.isfinite(value[k]) for k in ("actual_goal_mse", "refreshed_goal_mse",
                                                         "open_rgb_mse", "refreshed_rgb_mse")), "finite metrics")
                rows.append(value)
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(before == after and all(sha(p) == d for p, d in sources.items()), "model/source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "excluded": excluded,
                  "fixed_rows": 16, "host": host, "sources": sources, "model_before": before,
                  "model_after": after, "calls_code_path": {"encode": 3 * len(rows),
                    "predict": 3 * len(rows), "action_encoder": 3 * len(rows)},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Known roots; midpoint changes observation and horizon; no post-stop imputation."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "excluded": len(excluded), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replay actual completed actions and encode actual cp5/midpoint/terminal RGB on CUDA.", flush=True)
    run(args.output)
