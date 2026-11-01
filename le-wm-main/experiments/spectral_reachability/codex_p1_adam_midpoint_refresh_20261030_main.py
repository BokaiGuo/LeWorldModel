"""Same executed Adam packets: open prediction versus real-RGB midpoint refresh."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_rgb_gap_20261030_main as gap
from . import codex_p1_same_first5_feedback_compare_20261030_main as trace
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback

base = gap.base
NAME = "codex_p1_adam_midpoint_refresh_20261030_main"
OPEN = base.ROOT / "results/codex_p1_adam_rgb_gap_20261030_main_RUN01.json"
OPEN_SHA = "264d1aaf155896b7ead155624d1ffb074d829be6cee13da80e116500700c48e8"


def inputs():
    old, _ = gap.prior.gap.read(OPEN, OPEN_SHA)
    report, _ = gap.prior.gap.read(gap.PHYSICAL / "run_report.json", gap.REPORT_SHA)
    entries = gap.inputs()
    if len(old["rows"]) != len(entries):
        raise ValueError("denominator")
    result = []
    for entry, saved in zip(entries, old["rows"], strict=True):
        i, row, physical, plan, raw, terminal, last_frame = entry
        if saved["index"] != i or saved["action_sha256"] != physical["raw10_sha256"]:
            raise ValueError("open prediction identity")
        folder = gap.PHYSICAL / f"row_{i:02d}"
        shard = report["shards"][i]
        ts = trace.transitions(folder, shard["report_sha256"])
        if len(ts) != 10 or any(t["step"] != j + 1 or t["actual_bits"] != raw.reshape(10, 2)[j].tobytes().hex()
                                for j, t in enumerate(ts)):
            raise ValueError("executed action mismatch")
        item, _ = gap.prior.gap.read(folder / "report.json", shard["report_sha256"])
        frames = [f for f in item["row"]["trajectory"] if f["phase"] == "probe" and f["step"] == 5
                  and f["condition"] == physical["condition"]
                  and f["overlay_index"] == (1 if physical["condition"] == "on" else 0)]
        if len(frames) != 1:
            raise ValueError("midpoint frame count")
        frame = frames[0]
        path = folder / frame["png"]
        if (Path(frame["png"]).name != frame["png"] or base.sha(path) != frame["file_sha256"]
                or frame["state"] != ts[4]["state"]):
            raise ValueError("midpoint frame binding")
        with Image.open(path) as image:
            mid = np.asarray(image).copy()
        if mid.shape != (224, 224, 3) or hashlib.sha256(mid.tobytes()).hexdigest() != frame["pixel_sha256"]:
            raise ValueError("midpoint pixels")
        result.append((entry, saved, mid, frame))
    return result


def run(out):
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
        tx = p0.image_transform()
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry, old, rgb, frame in entries:
                i, row, physical, plan, raw, terminal, last_frame = entry
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_rgb = np.asarray(image).copy()
                encode = lambda x: model.encode({"pixels": p0._image_tensor(x, transform=tx, device=device)})["emb"][0, 0]
                cp5, midpoint, actual = encode(cp5_rgb), encode(rgb), encode(terminal)
                states, history, goal, mean, std = feedback.second_context(row, cp5, midpoint, raw[0])
                token = ((torch.as_tensor(raw[1], device=device) - mean) / std).reshape(1, 1, 10)
                pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[0, -1]
                mse = lambda a, b: float((a-b).square().mean().item())
                actual_diff = float((actual - torch.tensor(old["actual_latent"], device=device)).abs().max().item())
                if actual_diff > 1e-5 or not bool(torch.isfinite(pred).all()):
                    raise ValueError("terminal re-encoding or prediction mismatch")
                rows.append({"index": i, "logical_pair": physical["logical_pair"], "role": physical["role"],
                             "root_slot": physical["root_slot"], "condition": physical["condition"],
                             "action_sha256": physical["raw10_sha256"], "mid_pixel_sha256": frame["pixel_sha256"],
                             "open_error_mse": old["prediction_error_mse"],
                             "refresh_error_mse": mse(pred, actual), "refresh_goal_mse": mse(pred, goal[0]),
                             "actual_goal_mse": mse(actual, goal[0]),
                             "refreshed_latent": pred.cpu().tolist(), "actual_latent": actual.cpu().tolist(),
                             "actual_reencode_max_diff": actual_diff})
        torch.cuda.synchronize(device)
        if before != base.model_hash(model):
            raise ValueError("model mutated")
    result = {"schema": NAME, "status": "complete", "rows": rows, "fixed_roster": 24,
              "omitted_native_stopped": 5, "host": host, "model_before": before, "model_after": before,
              "open_sha256": OPEN_SHA, "code_sha256": base.sha(Path(__file__).resolve()),
              "explicit_calls": {"encoder": 57, "action_encoder": 19, "predict": 19},
              "P1_admission": False, "independent_confirmation": False,
              "limitations": "post-outcome known roots and duplicate trajectories; open vs refresh differs in "
                             "prediction horizon; no online replanning, budget or physical control evidence"}
    print(json.dumps({"rows": len(rows), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != base.ROOT / "results" or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    print("Replay identical executed Adam packets using real midpoint RGB, host CUDA only.", flush=True)
    run(out)
