"""Reconstruct prediction vectors for fixed P1 cp5 actions on host CUDA.

Diagnostic only: known roots and post-outcome action choices, no policy selection.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap

NAME = "codex_p1_cp5_vector_replay_20261030_main"
RGB = base.ROOT / "results/codex_p1_cp5_terminal_rgb_gap_20261030_main_RUN01.json"
RGB_SHA = "f00df372bc3ee70ea6c02ed6b08f609ee318897333a83616c806d1352674c85d"


def inputs():
    paired, _ = gap.read(gap.PAIRS, gap.PAIR_SHA)
    entries = gap.terminal_inputs(paired)
    rgb, _ = gap.read(RGB, RGB_SHA)
    bundle, _ = gap.read(base.BUNDLE, base.BUNDLE_SHA)
    if len(entries) != 48 or len(rgb["rows"]) != 48 or rgb["status"] != "complete":
        raise ValueError("complete fixed roster required")
    for entry, saved in zip(entries, rgb["rows"], strict=True):
        if any(entry[k] != saved[k] for k in ("index", "label", "event_sha256", "frame_sha256",
                                                "pixel_sha256", "predicted_goal_mse", "identity")):
            raise ValueError("RGB result identity mismatch")
        actual = np.asarray(saved["actual_latent"], dtype=np.float32)
        if actual.shape != (192,) or not np.isfinite(actual).all():
            raise ValueError("actual latent")
        entry["actual_latent"] = actual
    return bundle, entries


def replay(model, row, cp5, raw):
    """Literal two-packet version of the original final replay."""
    import torch

    states, history, goal, mean, std = base.bridge.cp5.context(row, cp5, device=cp5.device)
    for packet in range(2):
        norm = ((raw[:, packet] - mean) / std).reshape(1, 1, 10)
        context = torch.cat((history, norm), dim=1)
        prediction = model.predict(states, model.action_encoder(context))
        if prediction.shape != states.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError("prediction shape/finite")
        states = torch.cat((states, prediction[:, -1:]), dim=1)[:, -3:]
        history = context[:, -2:]
    return states[0, -1], goal[0]


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    bundle, entries = inputs()
    sources = base.bindings()
    sources.update({str(Path(__file__).resolve()): base.sha(Path(__file__).resolve()),
                    str(Path(gap.__file__).resolve()): base.sha(Path(gap.__file__).resolve()),
                    str(RGB): RGB_SHA, str(gap.PAIRS): gap.PAIR_SHA,
                    str(base.BUNDLE): base.BUNDLE_SHA})
    for entry in entries:
        for kind in ("event", "frame", "report"):
            sources[entry[kind + "_path"]] = entry[kind + "_sha256"]
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("frozen checkpoint mismatch")
        transform = p0.image_transform()
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for i in range(24):
                row = bundle["rows"][i]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    image_array = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(image_array, transform=transform,
                                                               device=device)})["emb"][0, 0]
                for entry in entries[2*i:2*i+2]:
                    event, _ = gap.read(Path(entry["event_path"]), entry["event_sha256"])
                    raw = torch.as_tensor(event["result"]["actions"], dtype=torch.float32, device=device)
                    pred, goal = replay(model, row, cp5, raw)
                    score = float((pred - goal).square().mean().item())
                    if abs(score - entry["predicted_goal_mse"]) > 1e-5:
                        raise ValueError("original replay score mismatch")
                    actual = torch.as_tensor(entry["actual_latent"], device=device)
                    rows.append({"index": i, "label": entry["label"], "identity": entry["identity"],
                                 "predicted_goal_mse": score,
                                 "actual_goal_mse": float((actual - goal).square().mean().item()),
                                 "prediction_vs_actual_mse": float((pred - actual).square().mean().item()),
                                 "predicted_latent": pred.cpu().tolist(),
                                 "actual_latent": entry["actual_latent"].tolist(),
                                 "physical_metrics": entry["physical_metrics"]})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": before,
                  "calls": {"cp5_encode": 24, "action_encoder": 96, "predict": 96},
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Known roots and repeated trajectories; observational replay, "
                                 "original row-score hashes not independently anchored."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Replay fixed physical actions and compare predicted vs real terminal latent.", flush=True)
    run(args.output)
