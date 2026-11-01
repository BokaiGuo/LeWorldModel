"""Encode fixed two/20-step physical terminal RGBs on host CUDA.

Descriptive known-root diagnosis, not a control experiment or P1 admission.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

ROOT = base.ROOT
NAME = "codex_p1_cp5_terminal_rgb_gap_20261030_main"
PAIRS = ROOT / "results/codex_p1_cp5_twenty_physical_saved_20261030_v1_CODEX01.json"
PAIR_SHA = "61860286e0a351cfd553fb5aaa5bd986924489bf2a9a6c66601ef2ec8da6147f"
RUNS = {
    "two_step": (ROOT / "results/codex_p1_cp5_physical_24_20261030_v1_run_CODEX01",
                 ROOT / "results/codex_p1_cp5_exploratory_cuda_20261030_v1_RUN01"),
    "twenty_step": (ROOT / "results/codex_p1_cp5_twenty_physical_20261030_v1_run_CODEX01",
                    ROOT / "results/codex_p1_cp5_twenty_step_cuda_20261030_main_RUN01"),
}


def read(path, expected=None):
    digest = base.sha(path)
    if expected is not None and digest != expected:
        raise ValueError("input hash mismatch: " + str(path))
    return json.loads(path.read_bytes()), digest


def terminal_inputs(pairs):
    """Bind literal action bytes and terminal image to consumed physical rows."""
    import hashlib

    if (pairs["status"] != "24_pairs_saved_verified" or len(pairs["rows"]) != 24
            or pairs["P1_admission"] is not False):
        raise ValueError("unverified paired roster")
    bundle, _ = read(base.BUNDLE, base.BUNDLE_SHA)
    reports = {}
    for label, (physical, _) in RUNS.items():
        reports[label], _ = read(physical / "run_report.json", pairs[
            "two_run_sha256" if label == "two_step" else "twenty_run_sha256"])
    entries = []
    for i, pair in enumerate(pairs["rows"]):
        if pair["index"] != i or bundle["rows"][i]["index"] != i:
            raise ValueError("row identity")
        for label, (physical, planning) in RUNS.items():
            shard = reports[label]["shards"][i]
            if shard["index"] != i:
                raise ValueError("shard identity")
            report_path = physical / f"row_{i:02d}" / "report.json"
            report, _ = read(report_path, shard["report_sha256"])
            row = report["row"]
            saved = pair[label]
            if (row["status"] != "probe_completed" or saved["status"] != "probe_completed"
                    or len(row["responses"]) != 10 or saved["accepted_suffix_steps"] != 10
                    or row["responses"][-1]["state"] != saved["horizon_endpoint_state"]):
                raise ValueError("complete endpoint required")
            frame = row["trajectory"][-1]
            if (frame["phase"] != "probe" or frame["step"] != 10
                    or frame["condition"] != pair["condition"]
                    or frame["overlay_index"] != (1 if pair["condition"] == "on" else 0)
                    or frame["state"] != saved["horizon_endpoint_state"]):
                raise ValueError("terminal frame identity")
            png = physical / f"row_{i:02d}" / frame["png"]
            if png.name != frame["png"] or base.sha(png) != frame["file_sha256"]:
                raise ValueError("terminal PNG hash")
            with Image.open(png) as image:
                rgb = np.asarray(image).copy()
            if (rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8
                    or hashlib.sha256(rgb.tobytes()).hexdigest() != frame["pixel_sha256"]):
                raise ValueError("terminal pixel hash")
            event_path = planning / f"row_{i:02d}_result.json"
            event, event_sha = read(event_path)
            result = event["result"]
            raw = np.asarray(result["actions"], dtype="<f4")
            if (event["index"] != i or result["index"] != i or raw.shape != (1, 2, 5, 2)
                    or hashlib.sha256(raw.tobytes()).hexdigest() != saved["raw10_sha256"]
                    or not np.isfinite(result["score"])):
                raise ValueError("planner action/score binding")
            entries.append({"index": i, "label": label, "rgb": rgb,
                            "event_sha256": event_sha, "frame_sha256": frame["file_sha256"],
                            "event_path": str(event_path), "frame_path": str(png),
                            "report_path": str(report_path), "report_sha256": shard["report_sha256"],
                            "pixel_sha256": frame["pixel_sha256"],
                            "predicted_goal_mse": result["score"],
                            "goal_latent": bundle["rows"][i]["goal_latent"],
                            "identity": {k: pair[k] for k in
                                         ("root_slot", "condition", "arm", "role", "logical_pair")},
                            "physical_metrics": saved["horizon_metrics"]})
    return entries


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    pair, pair_sha = read(PAIRS, PAIR_SHA)
    entries = terminal_inputs(pair)
    source = base.bindings()
    source[str(Path(__file__).resolve())] = base.sha(Path(__file__).resolve())
    for entry in entries:
        for kind in ("event", "frame", "report"):
            source[entry[kind + "_path"]] = entry[kind + "_sha256"]
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen model mismatch")
        transform = p0.image_transform()
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for entry in entries:
                pixels = p0._image_tensor(entry["rgb"], transform=transform, device=device)
                latent = model.encode({"pixels": pixels})["emb"][0, 0]
                goal = torch.tensor(entry["goal_latent"], dtype=torch.float32, device=device)
                if latent.shape != (192,) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("encoded latent")
                mse = float((latent - goal).square().mean().item())
                rows.append({k: v for k, v in entry.items() if k not in ("rgb", "goal_latent")} |
                            {"actual_rgb_goal_mse": mse, "actual_latent": latent.cpu().tolist()})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or base.sha(PAIRS) != pair_sha:
            raise ValueError("model or paired data drift")
        if any(base.sha(Path(path)) != digest for path, digest in source.items()):
            raise ValueError("source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "pair_sha256": pair_sha,
                  "sources": source, "host": host, "model_before": before, "model_after": before,
                  "encoder_calls": 48, "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Known roots; duplicate trajectories; scores not independently anchored "
                                 "in original report; model-goal MSE is not physical success."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Compare predicted and actual RGB latent goal MSE on fixed physical actions.", flush=True)
    run(args.output)
