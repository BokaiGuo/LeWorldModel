"""Replay all complete fixed budget trajectories with actual midpoint RGB."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_adam_terminal_rgb_cuda_20261031 as terminal
from . import codex_p1_same_first5_feedback_compare_20261030_main as trace
from . import codex_p1_two_stage_feedback_20261030_v1 as feedback

NAME = "p1_adam_budget_midpoint_cuda_20261031"
ROOT = terminal.ROOT
OPEN = ROOT / "results/p1_adam_terminal_rgb_cuda_20261031_RUN01.json"
OPEN_SHA = "b7e64fbdb226405ad09f1086091829ba87ca365627debe7749da60ca774c2540"
PROTOCOL = ROOT.parent / "docs/D2P_P1_ADAM_BUDGET_MIDPOINT_PROTOCOL_20261031_MAIN.md"
need = terminal.need


def inputs():
    entries, stopped, sources = terminal.collect()
    opened = terminal.pair.pairing.read(OPEN, OPEN_SHA)
    lookup = {(row["index"], row["step"]): row for row in opened["rows"]}
    need(len(lookup) == len(entries) == 64 and len(stopped) == 8, "open roster")
    sources[str(OPEN)] = OPEN_SHA
    sources[str(PROTOCOL)] = terminal.pair.physical.sha(PROTOCOL)
    sources[str(Path(__file__).resolve())] = terminal.pair.physical.sha(Path(__file__).resolve())
    result = []
    for entry in entries:
        i, step = entry["index"], entry["step"]
        saved = lookup[(i, step)]
        need(saved["action_sha256"] == entry["action_sha256"]
             and saved["pixel_sha256"] == entry["pixel_sha256"], "open identity")
        directory, run_sha = terminal.RUNS[step]
        run = terminal.pair.pairing.read(directory / "run_report.json", run_sha)
        shard = run["shards"][i]
        folder = directory / f"row_{i:02d}"
        report = terminal.pair.pairing.read(folder / "report.json", shard["report_sha256"])
        transitions = trace.transitions(folder, shard["report_sha256"])
        raw = entry["action"].reshape(10, 2)
        need(len(transitions) == 10 and all(
            t["step"] == j + 1 and t["actual_bits"] == raw[j].tobytes().hex()
            and t["state"] == report["row"]["responses"][j]["state"]
            for j, t in enumerate(transitions)), "executed action and state binding")
        journal = folder / "journal.jsonl"
        sources[str(journal)] = report["journal_sha256"]
        frames = [f for f in report["row"]["trajectory"]
                  if f["phase"] == "probe" and f["step"] == 5
                  and f["condition"] == report["row"]["trajectory"][-1]["condition"]
                  and f["overlay_index"] == report["row"]["trajectory"][-1]["overlay_index"]]
        need(len(frames) == 1 and frames[0]["state"] == transitions[4]["state"], "midpoint binding")
        frame = frames[0]
        path = folder / frame["png"]
        need(path.name == frame["png"] and terminal.pair.physical.sha(path) == frame["file_sha256"], "midpoint file")
        with Image.open(path) as image:
            rgb = np.asarray(image).copy()
        need(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
             and hashlib.sha256(rgb.tobytes()).hexdigest() == frame["pixel_sha256"], "midpoint pixels")
        sources[str(path)] = frame["file_sha256"]
        result.append((entry, saved, rgb, frame["pixel_sha256"]))
    return result, stopped, sources


def run(out):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    terminal.curve.planner.configure()
    base = terminal.curve.planner.adam.prior.base
    entries, stopped, sources = inputs()
    for path, digest in base.bindings().items():
        need(path not in sources or sources[path] == digest, "source collision")
        sources[path] = digest
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "model identity")
        tx = p0.image_transform()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry, old, midpoint_rgb, mid_sha in entries:
                row, raw = entry["row"], entry["action"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    cp_rgb = np.asarray(image).copy()
                def encode(rgb):
                    return model.encode({"pixels": p0._image_tensor(rgb, transform=tx, device=device)})["emb"][0, 0]
                cp, mid, actual = encode(cp_rgb), encode(midpoint_rgb), encode(entry["rgb"])
                need(float((actual - torch.tensor(old["actual_latent"], device=device)).abs().max()) <= 1e-5,
                     "terminal re-encoding")
                states, history, goal, mean, std = feedback.second_context(row, cp, mid, raw[0])
                token = ((torch.as_tensor(raw[1], device=device) - mean) / std).reshape(1, 1, 10)
                pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[0, -1]
                need(bool(torch.isfinite(pred).all()), "finite prediction")
                rows.append({"index": entry["index"], "step": entry["step"], "role": row["role"],
                             "logical_pair": row["logical_pair"], "action_sha256": entry["action_sha256"],
                             "mid_pixel_sha256": mid_sha, "open_error_mse": old["prediction_rgb_mse"],
                             "refresh_error_mse": float((pred-actual).square().mean()),
                             "open_goal_mse": old["predicted_goal_mse"],
                             "refresh_goal_mse": float((pred-goal[0]).square().mean()),
                             "actual_goal_mse": old["actual_rgb_goal_mse"],
                             "refresh_latent": pred.cpu().tolist(), "actual_latent": actual.cpu().tolist()})
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(before == after and all(base.sha(path) == digest for path, digest in sources.items()), "source drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "stopped": stopped,
                  "host": host, "sources": sources, "model_before": before, "model_after": after,
                  "calls_code_path": {"encode": 3*len(rows), "predict": len(rows), "action_encoder": len(rows)},
                  "P1_admission": False, "independent_confirmation": False,
                  "limitations": "Known roots and repeated trajectories; horizon and observation both differ; "
                                 "no online replanning or post-stop imputation."}
        digest = base.write(out, result)
    print(json.dumps({"rows": len(rows), "stopped": len(stopped), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Replay fixed completed budget trajectories with real midpoint RGB on host CUDA.", flush=True)
    run(Path(parser.parse_args().output))
