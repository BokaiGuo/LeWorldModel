"""Recompute Adam predicted and executed RGB goal distances on uncensored rows."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_multistart_cp5_20261030_main as prior
from . import codex_p1_adam_paired_join_20261030_main as joined

base = prior.base
NAME = "codex_p1_adam_rgb_gap_20261030_main"
PHYSICAL = base.ROOT / "results/codex_p1_adam_physical_20261030_main_RUN01"
REPORT_SHA = "5610659ec9b917e1c28c059aa72aa9105949ce0593abc0fe2e02fecbf39b3311"


def inputs():
    data = {name: prior.gap.read(base.ROOT / path, digest)[0]
            for name, (path, digest) in joined.SOURCES.items()}
    joined.build(**data)
    bundle, _ = prior.gap.read(base.BUNDLE, base.BUNDLE_SHA)
    report, _ = prior.gap.read(PHYSICAL / "run_report.json", REPORT_SHA)
    if len(report["shards"]) != 24 or len(bundle["rows"]) != 24:
        raise ValueError("fixed roster")
    entries = []
    for i, (p, s, q, b, shard) in enumerate(zip(
            data["prepare"]["roster"], data["saved"]["rows"], data["plan"]["rows"],
            bundle["rows"], report["shards"], strict=True)):
        if not (i == p["index"] == s["index"] == q["index"] == b["index"] == shard["index"]):
            raise ValueError("row order")
        if s["status"] != "probe_completed":
            continue
        folder = PHYSICAL / f"row_{i:02d}"
        item, _ = prior.gap.read(folder / "report.json", shard["report_sha256"])
        record = item["row"]
        frames = [x for x in record["trajectory"] if x["phase"] == "probe" and x["step"] == 10
                  and x["condition"] == s["condition"] and x["overlay_index"] == (s["condition"] == "on")]
        if len(frames) != 1 or len(record["responses"]) != 10:
            raise ValueError("terminal observation")
        f = frames[0]
        path = folder / f["png"]
        if Path(f["png"]).name != f["png"] or base.sha(path) != f["file_sha256"] or f["state"] != s["completed_endpoint"]:
            raise ValueError("terminal binding")
        with Image.open(path) as image:
            rgb = np.asarray(image).copy()
        if rgb.shape != (224, 224, 3) or hashlib.sha256(rgb.tobytes()).hexdigest() != f["pixel_sha256"]:
            raise ValueError("terminal RGB")
        raw = np.asarray(q["selected_actions"], dtype="<f4")
        if raw.shape != (2, 5, 2) or hashlib.sha256(raw.tobytes()).hexdigest() != s["raw10_sha256"]:
            raise ValueError("actions")
        entries.append((i, b, s, q, raw, rgb, f))
    if len(entries) != 19:
        raise ValueError("uncensored denominator")
    return entries


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    entries = inputs()
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("model identity")
        transform = p0.image_transform()
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for i, row, physical, plan, raw, rgb, frame in entries:
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                encode = lambda x: model.encode({"pixels": p0._image_tensor(x, transform=transform, device=device)})["emb"][0, 0]
                cp5, actual = encode(cp5_rgb), encode(rgb)
                pred, goal = prior.fixed.replay(model, row, cp5, torch.tensor(raw[None], device=device))
                mse = lambda a, b: float((a-b).square().mean().item())
                score = mse(pred, goal)
                if abs(score - plan["candidate_scores"][plan["selected_index"]]) > 1e-5:
                    raise ValueError("stored planner score mismatch")
                rows.append({"index": i, "logical_pair": physical["logical_pair"], "role": physical["role"],
                             "root_slot": physical["root_slot"], "condition": physical["condition"],
                             "action_sha256": physical["raw10_sha256"], "terminal_pixel_sha256": frame["pixel_sha256"],
                             "predicted_goal_mse": score, "actual_rgb_goal_mse": mse(actual, goal),
                             "prediction_error_mse": mse(pred, actual),
                             "predicted_latent": pred.cpu().tolist(), "actual_latent": actual.cpu().tolist(),
                             "physical_metrics": physical["endpoint_metrics"]})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model):
            raise ValueError("model mutated")
    result = {"schema": NAME, "status": "complete", "rows": rows, "fixed_roster": 24,
              "omitted_native_stopped": 5, "host": host, "model_before": before, "model_after": before,
              "source_hashes": joined.SOURCES, "physical_report_sha256": REPORT_SHA,
              "code_sha256": base.sha(Path(__file__).resolve()), "P1_admission": False,
              "limitations": "known roots, repeated actions and outcome-dependent exploratory optimizer; "
                             "only uncensored endpoints; does not validate training normalizer provenance"}
    print(json.dumps({"rows": len(rows), "sha256": base.write(output, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Replay frozen-model Adam actions and compare to real RGB on uncensored fixed rows.", flush=True)
    run(out)
