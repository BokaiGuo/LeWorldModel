"""Execute real midpoint CUDA replanning with a sealed offline parity guard."""
import argparse
import base64
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_response_midpoint_physical_20261031 as physical
from . import p1_response_cp5_same_action_20261031 as source

ROOT, need, sha = physical.ROOT, physical.need, physical.sha
NAME = "p1_response_midpoint_online_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_MIDPOINT_ONLINE_PARITY_PROTOCOL_20261031.md"


def build():
    fixed = physical.build()
    sources = dict(fixed["bindings"])
    for path in (PROTOCOL, Path(__file__).resolve()):
        sources[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in sources.items()), "source drift")
    return {"schema": NAME + ".prepare", "physical": fixed, "bindings": sources,
            "fixed_rows": 16, "P1_admission": False, "independent_confirmation": False}


def execute_online(anchor, item, entry, data, directory, factory, model, device, transform):
    import torch
    from . import preflight_pusht_d2p_p0 as p0

    base = physical.prior.engine.h10.base
    original_capture = base.capture
    events = []

    def live_capture(env, obs, phase, step, row, meter, store, reference=None):
        pixels = original_capture(env, obs, phase, step, row, meter, store, reference)
        if phase == "probe" and step == 5:
            receipt = row["trajectory"][-1]
            need(all(receipt[k] == item["midpoint"][k] for k in ("pixel_sha256", "state", "snapshot"))
                 and hashlib.sha256(pixels.tobytes()).hexdigest() == item["midpoint"]["pixel_sha256"],
                 "live midpoint mismatch")
            blind, raw = entry["blind"], entry["raw"]
            with Image.open(io.BytesIO(base64.b64decode(blind["cp5_png_base64"], validate=True))) as image:
                cp_pixels = np.asarray(image.convert("RGB")).copy()
            start = time.monotonic()
            with torch.no_grad():
                def encode(image):
                    return model.encode({"pixels": p0._image_tensor(
                        image, transform=transform, device=device)})["emb"][0, 0]
                cp, mid = encode(cp_pixels), encode(pixels)
                states, history, goal, mean, std = source.feedback.second_context(blind, cp, mid, raw[0])
            action = torch.nn.Parameter(torch.tensor(raw[1][None], dtype=torch.float32, device=device))
            optimizer = torch.optim.Adam([action], lr=0.05)

            def score():
                token = ((action - mean) / std).reshape(1, 1, 10)
                prediction = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
                return (prediction - goal).square().mean()

            with torch.no_grad():
                baseline = float(score())
            for _ in range(20):
                optimizer.zero_grad(set_to_none=True)
                loss = score()
                loss.backward()
                need(action.grad is not None and bool(torch.isfinite(action.grad).all()), "live gradient")
                optimizer.step()
                with torch.no_grad():
                    action.clamp_(-1, 1)
            with torch.no_grad():
                final = float(score())
            need(np.isfinite([baseline, final]).all(), "finite live score")
            chosen = action.detach().cpu().numpy()[0].astype("<f4") if final < baseline else raw[1].copy()
            expected = bytes.fromhex(item["raw10_float32_hex"])[40:]
            need(chosen.tobytes() == expected, "live versus sealed action mismatch")
            torch.cuda.synchronize(device)
            event = {"anchor": item["index"], "midpoint_pixel_sha256": receipt["pixel_sha256"],
                     "second_packet_hex": chosen.tobytes().hex(), "baseline_score": baseline,
                     "optimized_score": final, "optimizer_seconds": time.monotonic() - start,
                     "calls_code_path": {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20}}
            store.emit("online_midpoint_plan", **event)
            events.append(event)
        return pixels

    base.capture = live_capture
    try:
        report = physical.guarded_execute(anchor, item, data, directory, factory)
    finally:
        base.capture = original_capture
    if len(report["row"]["responses"]) >= 5:
        need(len(events) == 1, "missing online planning")
    return report, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-cuda-and-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    fixed = build()
    if args.command == "prepare":
        result = {"sha256": physical.prior.bundle.physical.write(out, fixed), "rows": 13}
    else:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from . import preflight_pusht_d2p_p0 as p0

        need(args.authorize_cuda_and_physics and args.prepared and args.sha256
             and sha(args.prepared) == args.sha256
             and physical.prior.bundle.physical.prior.native.read(args.prepared, cap=64 * 1024**2) == fixed
             and shutil.disk_usage(ROOT / "results").free >= 2 * 1024**3, "authorization/prepared/disk")
        entries, _, _ = source.collect()
        need([e["index"] for e in entries] == [a["index"] for a in fixed["physical"]["anchors"]],
             "live source roster")
        _, data = physical.prior.bundle.physical.pilot.material(physical.prior.bundle.physical.JOIN_SHA)
        physical.prior.bundle.physical.prior.native.constructor_contract()
        base = source.common.base
        with base.locking.gpu_lock(base.LOCK):
            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
            before = base.model_hash(model)
            need(before == base.MODEL_HASH, "frozen checkpoint")
            transform = p0.image_transform()
            out.mkdir(mode=0o700)

            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

            shards, online = [], []
            with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
                for anchor, item, entry in zip(fixed["physical"]["anchors"], fixed["physical"]["roster"],
                                                entries, strict=True):
                    directory = out / f"row_{anchor['index']:02d}"
                    report, events = execute_online(anchor, item, entry, data, directory, factory,
                                                    model, device, transform)
                    shards.append({"index": anchor["index"], "status": report["status"],
                                   "sha256": sha(directory / "report.json")})
                    online.extend(events)
                    print(json.dumps(shards[-1]), flush=True)
                    if report["status"] != "recorded_not_consumed":
                        break
            torch.cuda.synchronize(device)
            after = base.model_hash(model)
            need(before == after and all(sha(p) == h for p, h in fixed["bindings"].items()), "source/model drift")
            report = {"schema": NAME + ".run", "status": "recorded_not_consumed"
                      if len(shards) == 13 and all(x["status"] == "recorded_not_consumed" for x in shards)
                      else "partial_or_failed", "shards": shards, "online_events": online, "host": host,
                      "model_before": before, "model_after": after, "prepare_sha256": args.sha256,
                      "fixed_rows": 16, "P1_admission": False, "independent_confirmation": False}
            result = {"status": report["status"], "sha256": physical.prior.bundle.physical.write(out / "run_report.json", report)}
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
