"""Fixed-action predicted versus actual terminal RGB latent on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_multistart20_physical_20261030_main as physical
from . import codex_p1_cp5_vector_replay_20261030_main as prior

NAME = "codex_p1_multistart20_terminal_gap_20261030_main"
ROOT = prior.base.ROOT
RUN = ROOT/"results/codex_p1_multistart20_physical_20261030_main_RUN01"
RECEIPT = ROOT/"results/codex_p1_multistart20_physical_20261030_main_SAVED_RUN01.json"
RECEIPT_SHA = "c7cbdf2109d79cf29642659e2f751c8b20a75991cbb30e0427d2c4cdeea25454"
PLAN = physical.PLAN
PLAN_SHA = physical.PLAN_SHA


def inputs():
    base, gap = prior.base, prior.gap
    receipt, _ = gap.read(RECEIPT, RECEIPT_SHA)
    plan, _ = gap.read(PLAN, PLAN_SHA)
    bundle, _ = gap.read(base.BUNDLE, base.BUNDLE_SHA)
    report, _ = gap.read(RUN/"run_report.json", receipt["report_sha256"])
    prepared, _ = gap.read(RUN/"prepare.json", receipt["prepare_sha256"])
    if (receipt["status"] != "saved_checks_passed" or report["status"] != "recorded_not_consumed"
            or not all(len(x) == 24 for x in (receipt["rows"], plan["rows"], bundle["rows"],
                                               report["shards"], prepared["roster"]))):
        raise ValueError("complete roster")
    entries = []
    for i, (obs, p, row, shard, roster) in enumerate(zip(
            receipt["rows"], plan["rows"], bundle["rows"], report["shards"], prepared["roster"], strict=True)):
        if not (obs["index"] == p["index"] == row["index"] == shard["index"] == roster["index"] == i
                and obs["raw10_sha256"] == p["selected_action_sha256"] == roster["raw10_sha256"]
                and obs["role"] == p["role"] == row["role"] == roster["role"]
                and obs["logical_pair"] == p["logical_pair"] == row["logical_pair"] == roster["logical_pair"]):
            raise ValueError("identity mismatch")
        actions = np.asarray(p["selected_actions"], dtype="<f4")
        if (actions.shape != (2, 5, 2) or hashlib.sha256(actions.tobytes()).hexdigest() != obs["raw10_sha256"]
                or actions.tobytes().hex() != roster["raw10_float32_hex"]):
            raise ValueError("action mismatch")
        if obs["status"] != "probe_completed":
            if obs["completed_endpoint"] is not None:
                raise ValueError("stopped endpoint must be null")
            entries.append({"index": i, "status": obs["status"], "returned_suffix": obs["returned_suffix"]})
            continue
        path = RUN/f"row_{i:02d}"/"report.json"
        item, _ = gap.read(path, shard["report_sha256"])
        row_data = item["row"]
        frames = [f for f in row_data["trajectory"] if f["phase"] == "probe" and f["step"] == 10
                  and f["condition"] == obs["condition"]
                  and f["overlay_index"] == (1 if obs["condition"] == "on" else 0)]
        if (len(frames) != 1 or len(row_data["responses"]) != 10
                or frames[0]["state"] != obs["completed_endpoint"]
                or frames[0]["state"] != row_data["responses"][-1]["state"]):
            raise ValueError("frame state mismatch")
        frame = frames[0]
        png = path.parent/frame["png"]
        if png.name != frame["png"] or base.sha(png) != frame["file_sha256"]:
            raise ValueError("frame file mismatch")
        with Image.open(png) as image:
            pixels = np.asarray(image).copy()
        if (pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8
                or hashlib.sha256(pixels.tobytes()).hexdigest() != frame["pixel_sha256"]):
            raise ValueError("pixel mismatch")
        entries.append({"index": i, "status": obs["status"], "returned_suffix": obs["returned_suffix"],
                        "identity": {k: obs[k] for k in ("root_slot", "condition", "role", "logical_pair")},
                        "physical_metrics": obs["endpoint_metrics"], "frame_path": str(png),
                        "frame_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"],
                        "report_path": str(path), "report_sha256": shard["report_sha256"],
                        "actions": actions, "row": row, "pixels": pixels,
                        "saved_score": p["candidate_scores"][p["selected_index"]]})
    return entries


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = prior.base
    output = Path(output)
    if (not output.is_absolute() or output.parent != ROOT/"results" or
            not output.name.startswith(NAME+"_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    entries = inputs()
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(prior.__file__).resolve(), PLAN, RECEIPT,
                 RUN/"run_report.json", RUN/"prepare.json", base.BUNDLE):
        sources[str(path)] = base.sha(path)
    for entry in entries:
        if entry["status"] == "probe_completed":
            sources[entry["frame_path"]] = entry["frame_sha256"]
            sources[entry["report_path"]] = entry["report_sha256"]
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint mismatch")
        transform = p0.image_transform()
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for entry in entries:
                if entry["status"] != "probe_completed":
                    rows.append(entry)
                    continue
                row = entry["row"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_img = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(cp5_img, transform=transform, device=device)})["emb"][0, 0]
                actual = model.encode({"pixels": p0._image_tensor(entry["pixels"], transform=transform, device=device)})["emb"][0, 0]
                pred, goal = prior.replay(model, row, cp5, torch.as_tensor(entry["actions"][None], device=device))
                predicted_goal_mse = float((pred-goal).square().mean())
                if abs(predicted_goal_mse-entry["saved_score"]) > 1e-5:
                    raise ValueError("replay score mismatch")
                if any(x.shape != (192,) or not bool(torch.isfinite(x).all()) for x in (pred, actual, goal)):
                    raise ValueError("latent shape/finite")
                result = {k: v for k, v in entry.items() if k not in ("row", "pixels", "actions")}
                result.update(predicted_goal_mse=predicted_goal_mse,
                              actual_goal_mse=float((actual-goal).square().mean()),
                              predicted_actual_mse=float((pred-actual).square().mean()),
                              predicted_latent=pred.cpu().tolist(), actual_latent=actual.cpu().tolist(),
                              goal_latent=goal.cpu().tolist())
                rows.append(result)
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": before,
                  "calls": {"encode": 44, "action_encoder": 44, "predict": 44},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Two early native stops omitted, not imputed; known roots and duplicate "
                                 "trajectories; shared model path is not independent ABI validation."}
        print(json.dumps({"status": "complete", "rows": len(rows),
                          "sha256": base.write(output, result)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Compare fixed planned and real terminal latents for completed trajectories.", flush=True)
    run(args.output)
