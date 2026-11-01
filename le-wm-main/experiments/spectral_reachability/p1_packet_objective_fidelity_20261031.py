"""Replay packet-objective actions and compare same-time actual RGB on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_packet_objective_physical_20261031 as physical
from . import p1_uphill_packet_fidelity_20261031 as earlier
from . import codex_p1_h10_paired_table_20261030_main as metrics

base = earlier.base
ROOT = physical.physical.ROOT
NAME = "p1_packet_objective_fidelity_20261031"
PREP = ROOT / "results/p1_packet_objective_physical_20261031_prepare_RUN01.json"
PREP_SHA = "47c46d54bf2f1f511ab60488fc3eb3d9c0034b7b481e8e29471e48c3895c5079"
RUN = ROOT / "results/p1_packet_objective_physical_20261031_run_RUN01"
RUN_SHA = "c4ab3d99255d40c16759793867333f98f55da1f7741fc051e749bfb566f62d4c"
CONSUME = ROOT / "results/p1_packet_objective_physical_20261031_consume_RUN01.json"
CONSUME_SHA = "7fdc29a1c22179c2123bfa97ba392e9ad62c1657888715f2e395d6a0ab3aaed1"
PROTOCOL = ROOT.parent / "docs/D2P_P1_PACKET_OBJECTIVE_FIDELITY_PROTOCOL_20261031.md"


def inputs():
    read = earlier.prior.paired.read
    files = ((PREP, PREP_SHA), (RUN / "run_report.json", RUN_SHA),
             (CONSUME, CONSUME_SHA), (physical.PLAN, physical.PLAN_SHA),
             (physical.prior.planner.bundle.OUT, physical.prior.planner.BUNDLE_SHA),
             (physical.physical.bundle.PREP, physical.physical.PREP_SHA))
    prepared, report, consumed, plan, blind, original = [read(p, h) for p, h in files]
    need = physical.need
    need(report["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == PREP_SHA
         and len(prepared["roster"]) == len(report["shards"]) == len(consumed["rows"])
         == len(plan["rows"]) == len(blind["rows"]) == len(original["anchors"]) == 24,
         "fixed roster")
    sources = {str(p): h for p, h in files}
    entries = []
    for i, (item, shard, saved, proposal, row, anchor) in enumerate(zip(
            prepared["roster"], report["shards"], consumed["rows"], plan["rows"],
            blind["rows"], original["anchors"], strict=True)):
        need(i == item["index"] == shard["index"] == saved["index"] == proposal["index"]
             == row["index"] == anchor["index"]
             and item["role"] == saved["role"] == proposal["role"] == row["role"] == anchor["role"]
             and item["logical_pair"] == row["logical_pair"] == anchor["logical_pair"]
             and item["cp5_pixel_sha256"] == row["cp5_pixel_sha256"]
             and item["prefix5_float32_hex"] == row["prefix_raw_f32_hex"]
             and saved["status"] == "probe_completed" and saved["returned_suffix"] == 10,
             "row identity or completion")
        raw = np.asarray(proposal["selected_actions"], dtype="<f4")
        digest = hashlib.sha256(raw.tobytes()).hexdigest()
        need(raw.shape == (2, 5, 2) and np.isfinite(raw).all()
             and digest == item["raw10_sha256"] == proposal["selected_action_sha256"]
             and raw.tobytes().hex() == item["raw10_float32_hex"], "action identity")
        path = RUN / f"row_{i:02d}" / "report.json"
        artifact = read(path, shard["report_sha256"])
        executed = artifact["row"]
        need(artifact["raw10_sha256"] == digest and executed["status"] == saved["status"]
             and len(executed["responses"]) == 10
             and saved["completed_endpoint"] == executed["responses"][-1]["state"]
             and all(r["step"] == j and not r["terminated"] and not r["truncated"]
                     for j, r in enumerate(executed["responses"], 1)), "physical binding")
        sources[str(path)] = shard["report_sha256"]
        observations = []
        for step in (5, 10):
            frame = earlier.frame_at(executed, step, item["condition"])
            png = path.parent / frame["png"]
            need(png.parent == path.parent and base.sha(png) == frame["file_sha256"], "image path")
            with Image.open(png) as im:
                image = np.asarray(im.convert("RGB")).copy()
            need(image.shape == (224, 224, 3)
                 and hashlib.sha256(image.tobytes()).hexdigest() == frame["pixel_sha256"], "pixels")
            sources[str(png)] = frame["file_sha256"]
            observations.append({"pixels": image,
                                 "physical": metrics.physical_metrics(frame["state"], anchor["goal"])})
        entries.append({"index": i, "role": item["role"], "logical_pair": item["logical_pair"],
                        "root_slot": item["root_slot"], "condition": item["condition"],
                        "blind": row, "raw": raw, "observations": observations,
                        "selected_scores": proposal["candidate_packet_scores"][proposal["selected_index"]],
                        "action_sha256": digest})
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    physical.need(out.is_absolute() and out.parent == ROOT / "results"
                  and out.name.startswith(NAME + "_") and not out.exists() and not out.is_symlink(),
                  "exclusive output")
    entries, sources = inputs()
    for p, digest in base.bindings().items():
        physical.need(p not in sources or sources[p] == digest, "source collision")
        sources[p] = digest
    for path in (Path(__file__).resolve(), PROTOCOL, Path(earlier.__file__).resolve(),
                 Path(metrics.__file__).resolve()):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        physical.need(before == base.MODEL_HASH, "checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for e in entries:
                with Image.open(io.BytesIO(base64.b64decode(e["blind"]["cp5_png_base64"], validate=True))) as im:
                    cp_image = np.asarray(im.convert("RGB")).copy()
                def encode(image):
                    return model.encode({"pixels": p0._image_tensor(
                        image, transform=transform, device=device)})["emb"][0, 0]
                cp = encode(cp_image)
                states, history, goal, mean, std = base.bridge.cp5.context(e["blind"], cp, device=device)
                raw = torch.as_tensor(e["raw"], device=device)
                packets = []
                for j in range(2):
                    token = ((raw[j] - mean) / std).reshape(1, 1, 10)
                    context = torch.cat((history, token), dim=1)
                    prediction = model.predict(states, model.action_encoder(context))
                    physical.need(prediction.shape == states.shape and bool(torch.isfinite(prediction).all()),
                                  "prediction")
                    pred, actual = prediction[0, -1], encode(e["observations"][j]["pixels"])
                    predicted_goal = float((pred - goal[0]).square().mean())
                    physical.need(abs(predicted_goal - e["selected_scores"][j]) <= 1e-5, "planner score replay")
                    packets.append({"step": 5 * (j + 1), "predicted_goal_mse": predicted_goal,
                                    "actual_goal_mse": float((actual - goal[0]).square().mean()),
                                    "predicted_actual_mse": float((pred - actual).square().mean()),
                                    "predicted_latent": pred.cpu().tolist(),
                                    "actual_latent": actual.cpu().tolist(),
                                    "physical": e["observations"][j]["physical"]})
                    states = torch.cat((states, prediction[:, -1:]), dim=1)[:, -3:]
                    history = context[:, -2:]
                rows.append({k: e[k] for k in ("index", "role", "logical_pair", "root_slot",
                                                "condition", "action_sha256")}
                            | {"goal_latent": goal[0].cpu().tolist(), "packets": packets})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        physical.need(before == after and all(base.sha(p) == h for p, h in sources.items()), "drift")
        value = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                 "host": host, "model_before": before, "model_after": after,
                 "P1_admission": False, "independent_confirmation": False,
                 "calls": {"encode": 72, "predict": 48, "action_encoder": 48},
                 "limitations": "Known dependent roots; retrospective fixed-action replay; "
                                "no proof of training provenance or model-interface independence."}
        digest = base.write(out, value)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare all fixed packet-objective predictions and same-time physical RGB on CUDA.", flush=True)
    run(parser.parse_args().output)
