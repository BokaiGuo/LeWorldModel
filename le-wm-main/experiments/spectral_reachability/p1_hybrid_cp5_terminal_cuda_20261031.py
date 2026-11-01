"""Replay the two fixed cp5 plans and encode their physical terminal RGB on CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_hybrid_cp5_physical_20261031 as physical
from . import p1_hybrid_cp5_physical_saved_20261031 as saved
from . import codex_p1_multistart_cp5_20261030_main as common

ROOT, need, sha = physical.ROOT, physical.need, physical.sha
NAME = "p1_hybrid_cp5_terminal_cuda_20261031"
DIRECTORY = ROOT / "results/p1_hybrid_cp5_physical_20261031_RUN01"
REPORT_SHA = "16e520bdbb664fe80a91cae030c414c73d84b86ad2b0698e20b3840d0376a67d"
RECEIPT = ROOT / "results/p1_hybrid_cp5_physical_saved_20261031_RUN01.json"
RECEIPT_SHA = "70068a853c359b37d74dcac17f0121e9978b7227e4c85373582cb74a3b9764ff"
PREPARE = ROOT / "results/p1_hybrid_cp5_physical_20261031_prepare_RUN01.json"
PREPARE_SHA = "b91993a25e7c504d16015f0509685ef76566e9655d451a92918c10f06d6b243a"
base = common.base


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    verified = saved.consume(DIRECTORY, REPORT_SHA, PREPARE, PREPARE_SHA)
    need(sha(RECEIPT) == RECEIPT_SHA and physical.h.read(RECEIPT) == verified, "verified physical receipt")
    prepared, bundle = physical.h.read(PREPARE), physical.h.read(physical.BUNDLE)
    plan = physical.h.read(physical.PLAN)
    report = physical.h.read(DIRECTORY / "run_report.json")
    sources = dict(plan["sources"])
    for path in (PREPARE, RECEIPT, DIRECTORY / "run_report.json", Path(__file__).resolve(),
                 Path(saved.__file__).resolve(), Path(common.fixed.__file__).resolve()):
        sources[str(path)] = sha(path)
    inputs = []
    for anchor, row, b, action, shard in zip(prepared["anchors"], verified["rows"], bundle["rows"],
                                               plan["rows"], report["shards"], strict=True):
        need(row["status"] == "probe_completed" and row["returned_suffix"] == 10
             and anchor["index"] == row["index"] == shard["index"], "full terminal roster")
        folder = DIRECTORY / f"row_{anchor['index']:02d}"
        value = physical.h.read(folder / "report.json")
        journal = folder / "journal.jsonl"
        need(sha(folder / "report.json") == shard["sha256"]
             and sha(journal) == value["journal_sha256"], "physical source hash")
        events = saved.saved.events(journal, value["journal_chain_tail"])
        frames = [e["receipt"] for e in events if e["kind"] == "fresh_observation"
                  and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 10
                  and e["receipt"]["overlay_index"] == 0]
        need(len(frames) == 1 and frames[0]["state"] == row["completed_endpoint"], "terminal RGB identity")
        path = folder / frames[0]["png"]
        need(not path.is_symlink() and sha(path) == frames[0]["file_sha256"], "terminal image SHA")
        sources.update({str(folder / "report.json"): shard["sha256"], str(journal): value["journal_sha256"],
                        str(path): frames[0]["file_sha256"]})
        inputs.append((b["blind"], action, path, frames[0]))
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen model")
        transform, rows = p0.image_transform(), []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for blind, action, path, frame in inputs:
                with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as im:
                    cp5_rgb = np.asarray(im).copy()
                with Image.open(path) as im:
                    terminal_rgb = np.asarray(im).copy()
                need(hashlib.sha256(terminal_rgb.tobytes()).hexdigest() == frame["pixel_sha256"],
                     "terminal pixels")
                cp5 = model.encode({"pixels": p0._image_tensor(cp5_rgb, transform=transform,
                                                               device=device)})["emb"][0, 0]
                actual = model.encode({"pixels": p0._image_tensor(terminal_rgb, transform=transform,
                                                                  device=device)})["emb"][0, 0]
                raw_np = np.asarray(action["actions"], dtype="<f4")
                need(hashlib.sha256(raw_np.tobytes()).hexdigest() == action["actions_sha256"], "planned actions")
                pred, goal = common.fixed.replay(model, blind, cp5, torch.tensor(raw_np[None], device=device))
                prediction_cost = float((pred - goal).square().mean())
                need(abs(prediction_cost - action["candidate_scores"][action["selected_index"]]) <= 1e-5,
                     "selected planner replay")
                rows.append({"root_slot": action["root_slot"], "status": "complete",
                             "prediction_goal_mse": prediction_cost,
                             "real_rgb_goal_mse": float((actual - goal).square().mean()),
                             "prediction_actual_mse": float((pred - actual).square().mean()),
                             "terminal_pixel_sha256": frame["pixel_sha256"],
                             "action_sha256": action["actions_sha256"]})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(sha(p) == digest for p, digest in sources.items()),
             "model/source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "host": host,
                  "model_before": before, "model_after": before, "sources": sources,
                  "calls": {"encoder": 4, "action_encoder": 4, "predict": 4},
                  "physical_execution": False, "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Two known roots; fixed actions and terminal time only; no causal estimate."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "sha256": digest, "rows": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
