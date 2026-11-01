"""Recompute fixed imagined and real midpoint latents on host CUDA."""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import codex_p1_predicted_midpoint_10x2_20261030_main as pred
from . import codex_p1_predicted_vs_real_10x2_20261030_main as compare

NAME = "codex_p1_predicted_midpoint_replay_20261030_main"
base = pred.launcher.base


def run(out):
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0
    pair = compare.pair
    saved = pair.read(compare.PRED_SAVED, compare.PRED_SHA)
    bundle = pair.read(base.BUNDLE, base.BUNDLE_SHA)
    pair.need(len(saved["rows"]) == len(bundle["rows"]) == 24, "roster")
    results = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        pair.need(before == base.MODEL_HASH, "checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for i, (receipt, row) in enumerate(zip(saved["rows"], bundle["rows"], strict=True)):
                pair.need(receipt["index"] == row["index"] == i and len(receipt["stage_resources"]) == 2,
                          "complete stage roster")
                folder = compare.PRED_DIR/f"row_{i:02d}"
                extra = pair.read(folder/"stage_2/predicted_midpoint.json", receipt["midpoint_sidecar_sha256"])
                stage1 = json.loads((folder/"stage_1/report.json").read_bytes())
                raw = bytes.fromhex(stage1["result"]["raw_horizon_f32_hex"])
                packet = np.frombuffer(raw[:40], dtype="<f4").copy().reshape(5, 2)
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(cp5_rgb, transform=transform, device=device)})["emb"][0, 0]
                predicted = pred.predict_midpoint(row, cp5, packet, model)
                # The trusted physical journal reader validates the source frame in the saved consumer.
                report = json.loads((folder/"physical/report.json").read_bytes())
                trace = report["row"]["trajectory"]
                frames = [f for f in trace if f["phase"] == "probe" and f["step"] == 5
                          and f["condition"] == receipt["condition"]
                          and f["overlay_index"] == (1 if receipt["condition"] == "on" else 0)]
                pair.need(len(frames) == 1, "midpoint frame")
                frame = frames[0]
                path = folder/"physical"/frame["png"]
                pair.need(base.sha(path) == frame["file_sha256"], "frame hash")
                with Image.open(path) as image:
                    rgb = np.asarray(image).copy()
                actual = model.encode({"pixels": p0._image_tensor(rgb, transform=transform, device=device)})["emb"][0, 0]
                p = np.asarray(extra["predicted_latent"], dtype=np.float32)
                a = np.asarray(extra["actual_latent"], dtype=np.float32)
                pdiff = float(np.max(np.abs(predicted.cpu().numpy()-p)))
                adiff = float(np.max(np.abs(actual.cpu().numpy()-a)))
                pair.need(pdiff < 1e-5 and adiff < 1e-5, "midpoint numerical mismatch")
                results.append({"index": i, "predicted_max_abs_error": pdiff,
                                "actual_max_abs_error": adiff, "frame_sha256": frame["file_sha256"]})
        torch.cuda.synchronize(device)
        pair.need(base.model_hash(model) == before, "model changed")
    result = {"schema": NAME, "status": "complete", "rows": results, "host": host,
              "model_before": before, "model_after": before,
              "code_sha256": base.sha(Path(__file__).resolve()),
              "sources": {str(compare.PRED_SAVED): compare.PRED_SHA,
                          str(base.BUNDLE): base.BUNDLE_SHA, str(compare.PRED_DIR/"run_report.json"): compare.PRED_REPORT},
              "explicit_calls": {"encoder": 48, "projector": 48, "action_encoder": 24, "predict": 24},
              "limitations": "same checkpoint and model implementation, no independent ABI or training provenance",
              "P1_admission": False}
    print(json.dumps({"rows": len(results), "sha256": base.write(out, result),
                      "max_predicted_abs_error": max(r["predicted_max_abs_error"] for r in results),
                      "max_actual_abs_error": max(r["actual_max_abs_error"] for r in results)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    out = Path(ap.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT/"results" or not out.name.startswith(NAME+"_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    run(out)
