"""Compare fixed-action open and observed-midpoint predictions on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_cp5_vector_replay_20261030_main as prior

NAME = "codex_p1_cp5_midframe_replay_20261030_main"


def midpoint(entry):
    report, _ = gap.read(Path(entry["report_path"]), entry["report_sha256"])
    row = report["row"]
    matches = [f for f in row["trajectory"] if f["phase"] == "probe" and f["step"] == 5
               and f["condition"] == entry["identity"]["condition"]
               and f["overlay_index"] == (1 if entry["identity"]["condition"] == "on" else 0)]
    if len(matches) != 1 or row["responses"][4]["step"] != 5:
        raise ValueError("missing or ambiguous midpoint")
    frame = matches[0]
    if frame["state"] != row["responses"][4]["state"]:
        raise ValueError("midpoint state mismatch")
    path = Path(entry["report_path"]).parent / frame["png"]
    if path.name != frame["png"] or base.sha(path) != frame["file_sha256"]:
        raise ValueError("midpoint file hash mismatch")
    with Image.open(path) as image:
        rgb = np.asarray(image).copy()
    if (rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8
            or hashlib.sha256(rgb.tobytes()).hexdigest() != frame["pixel_sha256"]):
        raise ValueError("midpoint pixels mismatch")
    return rgb, path, frame


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if (not output.is_absolute() or output.parent != base.ROOT / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output required")
    bundle, entries = prior.inputs()
    mids = [midpoint(entry) for entry in entries]
    sources = base.bindings()
    for path in (Path(__file__).resolve(), Path(prior.__file__).resolve(),
                 Path(gap.__file__).resolve(), prior.RGB, gap.PAIRS, base.BUNDLE):
        sources[str(path)] = base.sha(path)
    for entry, (_, path, frame) in zip(entries, mids, strict=True):
        sources[str(path)] = frame["file_sha256"]
        for kind in ("event", "frame", "report"):
            sources[entry[kind + "_path"]] = entry[kind + "_sha256"]
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
            for entry, (rgb, path, frame) in zip(entries, mids, strict=True):
                row = bundle["rows"][entry["index"]]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp5_image = np.asarray(image).copy()
                cp5 = model.encode({"pixels": p0._image_tensor(cp5_image, transform=transform, device=device)})["emb"][0, 0]
                mid = model.encode({"pixels": p0._image_tensor(rgb, transform=transform, device=device)})["emb"][0, 0]
                event, _ = gap.read(Path(entry["event_path"]), entry["event_sha256"])
                raw = torch.as_tensor(event["result"]["actions"], dtype=torch.float32, device=device)
                states, history, goal, mean, std = base.bridge.cp5.context(row, cp5, device=device)
                context1 = torch.cat((history, ((raw[:, 0] - mean) / std).reshape(1, 1, 10)), dim=1)
                pred5 = model.predict(states, model.action_encoder(context1))[:, -1:]
                context2 = torch.cat((context1[:, -2:], ((raw[:, 1] - mean) / std).reshape(1, 1, 10)), dim=1)
                open_states = torch.cat((states, pred5), dim=1)[:, -3:]
                observed_states = torch.cat((states, mid.reshape(1, 1, -1)), dim=1)[:, -3:]
                encoded_action = model.action_encoder(context2)
                open10 = model.predict(open_states, encoded_action)[0, -1]
                refreshed10 = model.predict(observed_states, encoded_action)[0, -1]
                actual10 = torch.as_tensor(entry["actual_latent"], device=device)
                if abs(float((open10-goal[0]).square().mean()) - entry["predicted_goal_mse"]) > 1e-5:
                    raise ValueError("original score mismatch")
                vectors = {"pred5": pred5[0, 0], "actual5": mid, "open10": open10,
                           "refreshed10": refreshed10, "actual10": actual10}
                if any(v.shape != (192,) or not bool(torch.isfinite(v).all()) for v in vectors.values()):
                    raise ValueError("invalid latent")
                mse = lambda a, b: float((a-b).square().mean().item())
                rows.append({"index": entry["index"], "label": entry["label"], "identity": entry["identity"],
                             "midpoint_frame": str(path), "midpoint_sha256": frame["file_sha256"],
                             "midpoint_pixel_sha256": frame["pixel_sha256"], "midpoint_state": frame["state"],
                             "first_packet_mse": mse(pred5[0, 0], mid), "open_terminal_mse": mse(open10, actual10),
                             "refreshed_terminal_mse": mse(refreshed10, actual10),
                             "vectors": {k: v.cpu().tolist() for k, v in vectors.items()}})
        torch.cuda.synchronize(device)
        if base.model_hash(model) != before or any(base.sha(Path(p)) != h for p, h in sources.items()):
            raise ValueError("model or source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources,
                  "host": host, "model_before": before, "model_after": before,
                  "calls": {"encoder": 96, "action_encoder": 96, "predict": 144},
                  "independent_confirmation": False, "P1_admission": False,
                  "limitations": "Known roots and repeated trajectories; observed midpoint is a diagnostic "
                                 "oracle, not a physical feedback control result; planner score hashes were "
                                 "not independently anchored; dependency closure and normalizer provenance incomplete."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Compare fixed-action predictions with and without true midpoint RGB refresh.", flush=True)
    run(args.output)
