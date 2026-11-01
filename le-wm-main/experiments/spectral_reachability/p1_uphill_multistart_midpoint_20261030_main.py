"""Replay fixed P1 suffix actions with and without true midpoint RGB refresh."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import p1_uphill_multistart_rgb_20261030_main as rgb

NAME = "p1_uphill_multistart_midpoint_20261030_main"
RGB = base.ROOT / "results/p1_uphill_multistart_rgb_20261030_main_RUN01.json"
RGB_SHA = "1f5717ec7cf56bccaac0b05deb27785a33c5d532e74ec50321ba85b2a78c6c38"
PROTOCOL = base.ROOT.parent / "docs/D2P_P1_UPHILL_MULTISTART_MIDPOINT_PROTOCOL_20261030_MAIN.md"


def inputs():
    entries, sources = rgb.inputs()
    saved = rgb.paired.read(RGB, RGB_SHA)
    plan = rgb.paired.read(rgb.paired.physical.PLAN, rgb.paired.physical.PLAN_SHA)
    bundle = rgb.paired.read(rgb.paired.physical.planner.bundle.OUT, rgb.paired.physical.planner.BUNDLE_SHA)
    report = rgb.paired.read(rgb.RUN / "run_report.json", rgb.REPORT_SHA)
    if len(entries) != len(saved["rows"]) or saved["status"] != "complete":
        raise ValueError("complete RGB evidence")
    sources[str(RGB)] = RGB_SHA
    sources[str(rgb.paired.physical.PLAN)] = rgb.paired.physical.PLAN_SHA
    for entry, actual in zip(entries, saved["rows"], strict=True):
        i = entry["index"]
        if (any(entry[k] != actual[k] for k in ("index", "logical_pair", "role", "model_goal_mse", "pixel_sha256"))
                or len(actual["actual_latent"]) != 192 or not np.isfinite(actual["actual_latent"]).all()):
            raise ValueError("terminal RGB identity")
        report_path = rgb.RUN / f"row_{i:02d}" / "report.json"
        row = rgb.paired.read(report_path, report["shards"][i]["report_sha256"])["row"]
        frames = [f for f in row["trajectory"] if f["phase"] == "probe" and f["step"] == 5
                  and f["condition"] == row["condition"]
                  and f["overlay_index"] == (1 if row["condition"] == "on" else 0)]
        if len(frames) != 1 or frames[0]["state"] != row["responses"][4]["state"]:
            raise ValueError("midpoint state")
        frame = frames[0]
        path = report_path.parent / frame["png"]
        if path.name != frame["png"] or base.sha(path) != frame["file_sha256"]:
            raise ValueError("midpoint file")
        with Image.open(path) as image:
            midpoint = np.asarray(image).copy()
        if (midpoint.shape != (224, 224, 3) or midpoint.dtype != np.uint8
                or hashlib.sha256(midpoint.tobytes()).hexdigest() != frame["pixel_sha256"]):
            raise ValueError("midpoint pixels")
        raw = np.asarray(plan["rows"][i]["selected_actions"], dtype="<f4")
        if (raw.shape != (2, 5, 2) or hashlib.sha256(raw.tobytes()).hexdigest()
                != plan["rows"][i]["selected_action_sha256"]):
            raise ValueError("action bytes")
        entry.update(midpoint=midpoint, midpoint_pixel_sha256=frame["pixel_sha256"],
                     actual10=actual["actual_latent"], actions=raw, blind=bundle["rows"][i])
        sources[str(path)] = frame["file_sha256"]
    return entries, sources


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("output")
    entries, sources = inputs()
    sources.update(base.bindings())
    for path in (Path(__file__).resolve(), Path(rgb.__file__).resolve(), PROTOCOL):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        transform = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry in entries:
                row = entry["blind"]
                base.bridge.cp5.blind.validate_row(row)
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_image = np.asarray(image).copy()
                encode = lambda image: model.encode({"pixels": p0._image_tensor(
                    image, transform=transform, device=device)})["emb"][0, 0]
                cp5, mid = encode(cp_image), encode(entry["midpoint"])
                raw = torch.as_tensor(entry["actions"][None], dtype=torch.float32, device=device)
                states, history, goal, mean, std = base.bridge.cp5.context(row, cp5, device=device)
                ctx1 = torch.cat((history, ((raw[:, 0] - mean) / std).reshape(1, 1, 10)), dim=1)
                pred5 = model.predict(states, model.action_encoder(ctx1))[:, -1:]
                ctx2 = torch.cat((ctx1[:, -2:], ((raw[:, 1] - mean) / std).reshape(1, 1, 10)), dim=1)
                encoded = model.action_encoder(ctx2)
                open10 = model.predict(torch.cat((states, pred5), dim=1)[:, -3:], encoded)[0, -1]
                refresh10 = model.predict(torch.cat((states, mid.reshape(1, 1, -1)), dim=1)[:, -3:], encoded)[0, -1]
                actual10 = torch.as_tensor(entry["actual10"], dtype=torch.float32, device=device)
                mse = lambda x, y: float((x-y).square().mean().item())
                if abs(mse(open10, goal[0]) - entry["model_goal_mse"]) > 1e-5:
                    raise ValueError(f"planner score replay mismatch row {entry['index']}")
                vectors = {"pred5": pred5[0, 0], "actual5": mid, "open10": open10,
                           "refresh10": refresh10, "actual10": actual10}
                if any(v.shape != (192,) or not bool(torch.isfinite(v).all()) for v in vectors.values()):
                    raise ValueError("nonfinite vector")
                rows.append({"index": entry["index"], "logical_pair": entry["logical_pair"], "role": entry["role"],
                             "first_packet_mse": mse(pred5[0, 0], mid),
                             "open_terminal_mse": mse(open10, actual10),
                             "refreshed_terminal_mse": mse(refresh10, actual10),
                             "open_goal_mse": mse(open10, goal[0]),
                             "refreshed_goal_mse": mse(refresh10, goal[0]),
                             "actual_goal_mse": mse(actual10, goal[0]),
                             "midpoint_pixel_sha256": entry["midpoint_pixel_sha256"],
                             "vectors": {k: v.cpu().tolist() for k, v in vectors.items()}})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "host": host, "sources": sources,
                  "model_before": before, "model_after": after,
                  "calls": {"encoder": 2*len(rows), "action_encoder": 2*len(rows), "predict": 3*len(rows)},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Known dependent roots; offline observed midpoint, no real replanning; "
                                 "normalizer provenance and independent ABI are not certified."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Compare fixed open and observed-midpoint predictions on host CUDA.", flush=True)
    run(args.output)
