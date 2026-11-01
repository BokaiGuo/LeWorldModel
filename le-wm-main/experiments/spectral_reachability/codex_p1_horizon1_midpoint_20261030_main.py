"""Known-root, same-midpoint one-packet suffix optimization on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import codex_p1_cross_action_actual_mid_20261030_main as cross
from . import codex_p1_open10_matched_feedback_20261030_main as source

NAME = "codex_p1_horizon1_midpoint_20261030_main"
base, pair = cross.base, cross.pair


def run(output):
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    source.feedback.configure()
    checked = source.feedback.saved.consume(
        str(source.FB_DIR), source.FB_REPORT_SHA, str(source.FB_PREP), source.FB_PREP_SHA)
    report = pair.read(source.FB_DIR / "run_report.json", source.FB_REPORT_SHA)
    bundle = pair.read(base.BUNDLE, base.BUNDLE_SHA)
    pair.need(checked["status"] == "saved_checks_passed" and len(checked["rows"]) == 24
              and len(bundle["rows"]) == 24, "authenticated fixed roster")
    results = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        pair.need(before == base.MODEL_HASH, "checkpoint identity")
        tx = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH):
            for i, (item, row, shard) in enumerate(zip(
                    checked["rows"], bundle["rows"], report["shards"], strict=True)):
                pair.need(item["index"] == row["index"] == shard["index"] == i
                          and item["status"] == "probe_completed", "row identity and completion")
                folder = source.FB_DIR / f"row_{i:02d}"
                physical = pair.read(folder / "physical/report.json", shard["report_sha256"])
                transitions = pair.transitions(folder / "physical", shard["report_sha256"])
                pair.need(len(transitions) == 10, "ten executed actions")
                first = np.frombuffer(b"".join(bytes.fromhex(t["actual_bits"])
                                                for t in transitions[:5]), dtype="<f4").copy().reshape(5, 2)
                midpoint, mid_sha = cross.frame(folder, physical, item["condition"], 5)
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5 = np.asarray(image).copy()
                with torch.no_grad():
                    encode = lambda pixels: model.encode(
                        {"pixels": p0._image_tensor(pixels, transform=tx, device=device)})["emb"][0, 0]
                    z5, z10 = encode(cp5), encode(midpoint)
                    states, history, goal, mean, std = cross.pred.bridge.second_context(row, z5, z10, first)
                raw = torch.zeros((1, 1, 5, 2), device=device, dtype=states.dtype)
                torch.cuda.synchronize(device)
                started = time.monotonic()
                for _ in range(10):
                    action = raw.detach().requires_grad_(True)
                    token = ((action[:, 0] - mean) / std).reshape(1, 1, 10)
                    prediction = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))
                    loss = (prediction[:, -1] - goal).square().mean()
                    grad, = torch.autograd.grad(loss, action)
                    pair.need(bool(torch.isfinite(grad).all()), "finite gradient")
                    raw = (action - 0.05 * grad).clamp(-1, 1).detach()
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                packet = raw.cpu().numpy().astype("<f4", copy=False).reshape(5, 2)
                pair.need(np.isfinite(packet).all() and (np.abs(packet) <= 1).all(), "bounded result")
                results.append({"index": i, "first_packet_hex": first.tobytes().hex(),
                                "second_packet_hex": packet.tobytes().hex(),
                                "midpoint_pixel_sha256": mid_sha, "optimizer_seconds": elapsed})
        pair.need(base.model_hash(model) == before, "model mutation")
    result = {"schema": NAME, "status": "complete", "rows": results, "host": host,
              "model_before": before, "model_after": before,
              "code_sha256": base.sha(Path(__file__).resolve()),
              "sources": {str(source.FB_DIR / "run_report.json"): source.FB_REPORT_SHA,
                          str(source.FB_PREP): source.FB_PREP_SHA, str(base.BUNDLE): base.BUNDLE_SHA},
              "per_row_explicit_calls": {"encoder": 2, "projector": 2,
                                         "action_encoder": 10, "predict": 10, "backward": 10},
              "limitations": "known roots; explicit code-path counts, not hook-audited; "
                             "single-packet and historical two-packet planning have different forward costs",
              "independent_confirmation": False, "P1_admission": False}
    print(json.dumps({"rows": len(results), "sha256": base.write(output, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    pair.need(output.is_absolute() and output.parent == base.ROOT / "results"
              and output.name.startswith(NAME + "_") and not output.exists()
              and not output.is_symlink(), "exclusive output")
    print("Compare executed-horizon optimization using fixed actual midpoint inputs.", flush=True)
    run(output)
