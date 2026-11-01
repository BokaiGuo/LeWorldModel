"""Encode all uncensored fixed-horizon multistart physical endpoints on CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import p1_uphill_multistart_pairing_20261030_main as paired

NAME = "p1_uphill_multistart_rgb_20261030_main"
PAIR = base.ROOT / "results/p1_uphill_multistart_pairing_20261030_main_RUN01.json"
PAIR_SHA = "b69b9d38b7f926c31648dc00d0445e710a1014bdff2dff216a979c43ac397ff4"
RUN = base.ROOT / "results/p1_uphill_multistart_physical_20261030_main_run_RUN01"
REPORT_SHA = "623f23bb61d06ae105be4c0b34be6dabee1c7775b1fd5591e3791e53995a669c"
PROTOCOL = base.ROOT.parent / "docs/D2P_P1_UPHILL_MULTISTART_RGB_PROTOCOL_20261030_MAIN.md"


def inputs():
    pairs = paired.read(PAIR, PAIR_SHA)
    report = paired.read(RUN / "run_report.json", REPORT_SHA)
    bundle = paired.read(paired.physical.planner.bundle.OUT, paired.physical.planner.BUNDLE_SHA)
    observed = paired.read(paired.CONSUME, paired.CONSUME_SHA)
    if (len(pairs["rows"]) != 24 or len(report["shards"]) != 24
            or report["status"] != "recorded_not_consumed" or len(bundle["rows"]) != 24
            or len(observed["rows"]) != 24):
        raise ValueError("roster")
    entries, sources = [], {str(PAIR): PAIR_SHA, str(RUN / "run_report.json"): REPORT_SHA,
                            str(paired.CONSUME): paired.CONSUME_SHA,
                            str(paired.physical.planner.bundle.OUT): paired.physical.planner.BUNDLE_SHA}
    for i, (pair, shard, blind, physical) in enumerate(zip(
            pairs["rows"], report["shards"], bundle["rows"], observed["rows"], strict=True)):
        if not (pair["index"] == shard["index"] == blind["index"] == physical["index"] == i
                and pair["role"] == blind["role"] == physical["role"]
                and pair["status"] == physical["status"]):
            raise ValueError("row identity")
        if pair["status"] != "probe_completed":
            continue
        path = RUN / f"row_{i:02d}" / "report.json"
        row_report = paired.read(path, shard["report_sha256"])
        row = row_report["row"]
        frames = [x for x in row["trajectory"] if x["phase"] == "probe" and x["step"] == 10
                  and x["overlay_index"] == (1 if pair["condition"] == "on" else 0)]
        if (len(frames) != 1 or row["status"] != "probe_completed" or physical["returned_suffix"] != 10
                or frames[0]["state"] != physical["completed_endpoint"]):
            raise ValueError("fixed-horizon frame")
        frame = frames[0]
        png = path.parent / frame["png"]
        if png.name != frame["png"] or base.sha(png) != frame["file_sha256"]:
            raise ValueError("image binding")
        with Image.open(png) as image:
            rgb = np.asarray(image).copy()
        if (rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8
                or hashlib.sha256(rgb.tobytes()).hexdigest() != frame["pixel_sha256"]):
            raise ValueError("image pixels")
        sources[str(path)] = shard["report_sha256"]
        sources[str(png)] = frame["file_sha256"]
        entries.append({"index": i, "logical_pair": pair["logical_pair"], "role": pair["role"],
                        "model_goal_mse": pair["model_best_score"], "goal_latent": blind["goal_latent"],
                        "pixel_sha256": frame["pixel_sha256"], "rgb": rgb})
    if len(entries) != sum(r["status"] == "probe_completed" for r in pairs["rows"]):
        raise ValueError("censored denominator")
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
    for path in (Path(__file__).resolve(), PROTOCOL, Path(paired.__file__).resolve()):
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
                pixels = p0._image_tensor(entry["rgb"], transform=transform, device=device)
                latent = model.encode({"pixels": pixels})["emb"][0, 0]
                goal = torch.as_tensor(entry["goal_latent"], dtype=torch.float32, device=device)
                if latent.shape != (192,) or goal.shape != (192,) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("encoded latent")
                rows.append({k: v for k, v in entry.items() if k not in ("rgb", "goal_latent")}
                            | {"actual_goal_mse": float((latent-goal).square().mean()),
                               "actual_latent": latent.cpu().tolist()})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        if before != after or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows,
                  "fixed_roster": 24, "censored_rows": 24-len(rows), "encoder_calls": len(rows),
                  "sources": sources, "host": host, "model_before": before, "model_after": after,
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Known dependent roots, excludes early stopping at horizon, "
                                 "no physical success or causality implied by latent MSE."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "censored": 24-len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Encode all uncensored fixed-horizon physical RGB outcomes on host CUDA.", flush=True)
    run(args.output)
