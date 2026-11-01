"""Optimize a fixed second packet after real cp10 RGB on host CUDA."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_hybrid_cp5_midpoint_cuda_20261031 as source
from . import codex_p1_multistart_cp5_20261030_main as common

ROOT, need, sha = source.ROOT, source.need, source.sha
NAME = "p1_hybrid_cp10_feedback_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_HYBRID_CP10_FEEDBACK_EXPLORATION_PROTOCOL_20261031.md"
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
    verified = source.saved.consume(source.DIRECTORY, source.REPORT_SHA, source.PREPARE, source.PREPARE_SHA)
    need(sha(source.RECEIPT) == source.RECEIPT_SHA and source.physical.h.read(source.RECEIPT) == verified,
         "physical receipt")
    prep = source.physical.h.read(source.PREPARE)
    bundle = source.physical.h.read(source.physical.BUNDLE)
    plan = source.physical.h.read(source.physical.PLAN)
    report = source.physical.h.read(source.DIRECTORY / "run_report.json")
    sources = dict(plan["sources"])
    for path in (source.PREPARE, source.RECEIPT, source.DIRECTORY / "run_report.json", PROTOCOL,
                 Path(__file__).resolve(), Path(common.fixed.__file__).resolve()):
        sources[str(path)] = sha(path)
    inputs = []
    for anchor, b, action, shard in zip(prep["anchors"], bundle["rows"], plan["rows"],
                                         report["shards"], strict=True):
        folder = source.DIRECTORY / f"row_{anchor['index']:02d}"
        doc = source.physical.h.read(folder / "report.json")
        journal = folder / "journal.jsonl"
        need(sha(folder / "report.json") == shard["sha256"] and sha(journal) == doc["journal_sha256"],
             "physical source")
        events = source.saved.saved.events(journal, doc["journal_chain_tail"])
        frames = [e["receipt"] for e in events if e["kind"] == "fresh_observation"
                  and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5
                  and e["receipt"]["overlay_index"] == 0]
        need(len(frames) == 1, "unique physical midpoint")
        path = folder / frames[0]["png"]
        need(not path.is_symlink() and sha(path) == frames[0]["file_sha256"], "midpoint image")
        raw = np.asarray(action["actions"], dtype="<f4")
        need(hashlib.sha256(raw.tobytes()).hexdigest() == action["actions_sha256"]
             and action["logical_pair"] == anchor["logical_pair"], "fixed input action")
        sources.update({str(folder / "report.json"): shard["sha256"], str(journal): doc["journal_sha256"],
                        str(path): frames[0]["file_sha256"]})
        inputs.append((anchor, b["blind"], action, raw, path, frames[0]))
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen model")
        transform, rows = p0.image_transform(), []
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for anchor, blind, action, old_raw, path, frame in inputs:
                started = time.monotonic()
                with Image.open(path) as image:
                    pixels = np.asarray(image).copy()
                need(hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "pixel hash")
                with torch.no_grad():
                    cp10 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform,
                                                                    device=device)})["emb"][0, 0]
                # context() uses cp5 only as its last state. For cp10 we retain the
                # two real pre-cp10 states and the exact executed action history.
                import base64
                import io
                with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(cp5_rgb, transform=transform,
                                                                   device=device)})["emb"][0, 0]
                states, history, goal, mean, std = base.bridge.cp5.context(blind, cp5, device=device)
                states = torch.cat((states, cp10[None, None]), dim=1)[:, -3:]
                first = torch.tensor(old_raw[0], dtype=torch.float32, device=device)
                token = ((first - mean) / std).reshape(1, 1, 10)
                history = torch.cat((history, token), dim=1)[:, -2:]
                initial = common.initial_actions(anchor["logical_pair"])[:, 1]
                scores, candidates = [], []
                for init in initial:
                    current = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([current], lr=0.05)
                    for _ in range(20):
                        optimizer.zero_grad(set_to_none=True)
                        norm = ((current - mean) / std).reshape(1, 1, 10)
                        pred = model.predict(states, model.action_encoder(torch.cat((history, norm), 1)))[:, -1]
                        loss = (pred - goal).square().mean()
                        loss.backward()
                        need(current.grad is not None and bool(torch.isfinite(current.grad).all()), "gradient")
                        optimizer.step()
                        with torch.no_grad():
                            current.clamp_(-1, 1)
                    with torch.no_grad():
                        norm = ((current - mean) / std).reshape(1, 1, 10)
                        pred = model.predict(states, model.action_encoder(torch.cat((history, norm), 1)))[:, -1]
                        scores.append(float((pred - goal).square().mean()))
                    candidates.append(current.detach()[0].cpu().numpy().copy())
                winner = common.pick(scores)
                suffix = np.stack((old_raw[0], candidates[winner])).astype("<f4")
                need(np.isfinite(suffix).all() and (np.abs(suffix) <= 1).all()
                     and suffix[0].tobytes() == old_raw[0].tobytes(), "fixed first packet")
                torch.cuda.synchronize(device)
                rows.append({"index": anchor["index"], "root_slot": anchor["root_slot"],
                             "logical_pair": anchor["logical_pair"], "midpoint_pixel_sha256": frame["pixel_sha256"],
                             "prefix_raw_f32_hex": blind["prefix_raw_f32_hex"],
                             "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                             "candidate_scores": scores, "selected_index": winner, "actions": suffix.tolist(),
                             "actions_sha256": hashlib.sha256(suffix.tobytes()).hexdigest(),
                             "first_packet_sha256": hashlib.sha256(old_raw[0].tobytes()).hexdigest(),
                             "wall_seconds": time.monotonic() - started})
                print(json.dumps({"root_slot": anchor["root_slot"], "status": "planned"}), flush=True)
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(sha(p) == digest for p, digest in sources.items()),
             "model/source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": before,
                  "calls": {"encoder": 4, "action_encoder": 168, "predict": 168, "backward": 160},
                  "independent_confirmation": False, "P1_admission": False,
                  "physical_execution": False, "limitations": "Known roots, extra computation; no OS isolation."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
