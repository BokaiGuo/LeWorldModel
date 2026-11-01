"""Fixed-action cp10 CUDA precision sensitivity; no planning or physics."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_cp10_paired_prediction_cuda_20261031 as paired

ROOT, base, need, sha = paired.ROOT, paired.base, paired.need, paired.sha
NAME = "p1_cp10_precision_cuda_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_CP10_PRECISION_PROTOCOL_20261031.md"
REFERENCE = ROOT / "results/p1_cp10_paired_prediction_cuda_20261031_RUN01.json"
REFERENCE_SHA = "6dd3ee6f86aad55a963c065982f65d898f647ea287bebe6c659f3d198fe0a3cf"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith(NAME + "_") and output.suffix == ".json" and
         not output.exists() and not output.is_symlink(), "exclusive output")
    need(sha(REFERENCE) == REFERENCE_SHA, "reference SHA")
    reference = json.loads(REFERENCE.read_text())
    items, sources = paired.inputs()
    for path in (PROTOCOL, REFERENCE, Path(__file__).resolve()):
        sources[str(path)] = sha(path)
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        need(torch.cuda.is_bf16_supported(), "CUDA BF16 unsupported")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen model")
        transform, rows = p0.image_transform(), []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for item, prior in zip(items, reference["rows"], strict=True):
                need(item["anchor"]["root_slot"] == prior["root_slot"], "root binding")
                blind = item["blind"]
                with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as im:
                    cp5_rgb = np.asarray(im).copy()
                mid_path, mid_frame = item["frames"]["open"][0]
                cp10_rgb = paired.load_image(mid_path, mid_frame["pixel_sha256"])
                def encode(pixels):
                    return model.encode({"pixels": p0._image_tensor(
                        pixels, transform=transform, device=device)})["emb"][0, 0].float()
                with torch.autocast("cuda", enabled=False):
                    actual = {}
                    for label in ("open", "feedback"):
                        path, rec = item["frames"][label][1]
                        actual[label] = encode(paired.load_image(path, rec["pixel_sha256"]))
                        need(np.allclose(actual[label].cpu().numpy(), prior["arms"][label]["actual"],
                                         atol=1e-6, rtol=1e-6), "reference actual")
                modes = {}
                for mode in ("fp32", "bf16"):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(mode == "bf16")):
                        cp5, cp10 = encode(cp5_rgb), encode(cp10_rgb)
                        states, history, goal, mean, std = base.bridge.cp5.context(blind, cp5, device=device)
                        states = torch.cat((states, cp10[None, None]), dim=1)[:, -3:]
                        first = torch.tensor(item["open_raw"][0], dtype=torch.float32, device=device)
                        token = ((first - mean) / std).reshape(1, 1, 10)
                        history = torch.cat((history, token), dim=1)[:, -2:]
                        results = {}
                        for label in ("open", "feedback"):
                            raw = item[label + "_raw"][1]
                            norm = ((torch.tensor(raw, device=device) - mean) / std).reshape(1, 1, 10)
                            pred = model.predict(states, model.action_encoder(torch.cat((history, norm), 1)))[0, -1].float()
                            need(bool(torch.isfinite(pred).all()), "finite prediction")
                            if mode == "fp32":
                                need(np.allclose(pred.cpu().numpy(), prior["arms"][label]["prediction"],
                                                 atol=1e-6, rtol=1e-6), "reference prediction")
                            results[label] = {
                                "prediction": pred.cpu().tolist(),
                                "goal_mse": float((pred - goal[0]).square().mean()),
                                "actual_mse": float((pred - actual[label]).square().mean()),
                                "action_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
                    modes[mode] = results
                rows.append({"root_slot": prior["root_slot"], "modes": modes,
                             "real_goal_mse": {a: prior["arms"][a]["real_goal_mse"]
                                               for a in ("open", "feedback")}})
        torch.cuda.synchronize(device)
        need(base.model_hash(model) == before and all(sha(p) == v for p, v in sources.items()),
             "model/source drift")
        digest = base.write(output, {"schema": NAME, "status": "complete", "host": host,
                                    "model_before": before, "model_after": before, "rows": rows,
                                    "sources": sources,
                                    "calls": {"encoder": 12, "action_encoder": 8, "predict": 8},
                                    "independent_confirmation": False, "P1_admission": False})
    print(json.dumps({"status": "complete", "sha256": digest, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
