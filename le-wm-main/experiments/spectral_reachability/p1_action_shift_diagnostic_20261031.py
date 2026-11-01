"""Offline shifted-token controls against fixed five-step physical observations."""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_uphill_all_candidates_midpoint_20261031 as panel
from . import p1_candidate_short_horizon_alignment_20261031 as previous

base = panel.base
ROOT = base.ROOT
NAME = "p1_action_shift_diagnostic_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_ACTION_SHIFT_DIAGNOSTIC_PROTOCOL_20261031.md"


def action_variants(prefix, action):
    prefix, action = np.asarray(prefix, dtype="<f4"), np.asarray(action, dtype="<f4")
    if prefix.shape != (5, 2) or action.shape != (2, 5, 2) or \
            not np.isfinite(prefix).all() or not np.isfinite(action).all() or \
            np.abs(prefix).max() > 1 or np.abs(action).max() > 1:
        raise ValueError("raw action ABI")
    seq = action.reshape(10, 2)
    return {"correct": seq[:5].copy(),
            "earlier": np.concatenate((prefix[-1:], seq[:4]), axis=0),
            "later": seq[1:6].copy()}


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if not out.is_absolute() or out.parent != ROOT / "results" or \
            not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    entries, sources = panel.inputs()
    previous_result = panel.read(previous.MID, previous.MID_SHA)
    previous_rows = {(r["index"], r["kind"]): r for r in previous_result["rows"]}
    if previous_result["status"] != "complete" or len(previous_rows) != 96:
        raise ValueError("previous roster")
    sources.update(base.bindings())
    sources.update({str(previous.MID): previous.MID_SHA, str(PROTOCOL): base.sha(PROTOCOL),
                    str(Path(__file__).resolve()): base.sha(__file__)})
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint identity")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                row = entry["blind"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_pixels = np.asarray(image).copy()
                def encode(pixels):
                    return model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
                cp, actual = encode(cp_pixels), encode(entry["midpoint"])
                states, history, _, mean, std = base.bridge.cp5.context(row, cp, device=device)
                prefix = np.frombuffer(bytes.fromhex(row["prefix_raw_f32_hex"]), dtype="<f4").reshape(5, 2)
                variants = action_variants(prefix, entry["action"])
                costs = {}
                for name, action in variants.items():
                    raw = torch.as_tensor(action, dtype=torch.float32, device=device)
                    token = ((raw - mean) / std).reshape(1, 1, 10)
                    prediction = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))
                    if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError("prediction")
                    costs[name] = float((prediction[0, -1] - actual).square().mean())
                old = previous_rows[(entry["index"], entry["kind"])]
                if old["candidate"] != entry["candidate"] or \
                        abs(old["prediction_midpoint_mse"] - costs["correct"]) > 1e-5:
                    raise ValueError("saved prediction binding")
                rows.append({"index": entry["index"], "kind": entry["kind"], "candidate": entry["candidate"],
                             "role": row["role"], "logical_pair": row["logical_pair"],
                             "midpoint_pixel_sha256": entry["midpoint_pixel_sha256"],
                             "raw5_hex": {key: val.tobytes().hex() for key, val in variants.items()},
                             "raw5_sha256": {key: hashlib.sha256(val.tobytes()).hexdigest() for key, val in variants.items()},
                             "mse": costs})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(path) != digest for path, digest in sources.items()):
            raise ValueError("model or source drift")
        counts = {mode: dict(Counter(
            "lower" if r["mse"][mode] < r["mse"]["correct"] else
            "higher" if r["mse"][mode] > r["mse"]["correct"] else "equal" for r in rows))
                  for mode in ("earlier", "later")}
        digest = base.write(out, dict(schema=NAME, status="complete", host=host, rows=rows, counts=counts,
                                      sources=sources, model_before=before, model_after=after,
                                      encoder_calls=192, predictor_calls=288, action_encoder_calls=288,
                                      physical_execution=False, scientific_admission=False,
                                      independent_confirmation=False))
    print(json.dumps({"sha256": digest, "counts": counts,
                      "mean_mse": {k: float(np.mean([r["mse"][k] for r in rows]))
                                   for k in ("correct", "earlier", "later")}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Compare correct and one-step shifted action tokens with the same actual cp5 successor.", flush=True)
    run(parser.parse_args().output)
