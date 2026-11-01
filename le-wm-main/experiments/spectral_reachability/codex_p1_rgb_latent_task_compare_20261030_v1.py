"""Frozen CUDA RGB-distance diagnostic on every fixed multistart row.

No planning, training, physics, or post-outcome row selection. Native truth is
joined only after encoding. This is not a deployability or isolation proof.
"""
import argparse
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_p1_rgb_task_saved_compare_20261030_v2 as source
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as base

NAME = "codex_p1_rgb_latent_task_compare_20261030_v1"
ROOT = source.ROOT
RGB_SAVED = ROOT/"results/codex_p1_rgb_task_saved_compare_20261030_v2_CODEX01.json"
RGB_SHA = "93babd1bde41efc8cf1da2171dfe832bc496d5718c5fde1aa30a14f4d0124f53"
need = source.need
CAP, WALL = 48, 600


def compare_order(rows):
    result = []
    for slot, condition in itertools.product((0, 1), ("off", "on")):
        group = [r for r in rows if r["identity"]["root_slot"] == slot and r["identity"]["condition"] == condition]
        need(len(group) == 6, "fixed group")
        for left, right in itertools.combinations(group, 2):
            if not left["full_horizon"] or not right["full_horizon"]:
                result.append({"left": left["index"], "right": right["index"], "status": "early_stop"})
                continue
            a = left["latent_goal_distance"] - right["latent_goal_distance"]
            b = left["posthoc_pusher_error"] - right["posthoc_pusher_error"]
            kind = "tie" if a == 0 or b == 0 else ("agree" if a*b > 0 else "disagree")
            result.append({"left": left["index"], "right": right["index"], "status": kind,
                           "latent_distance_difference": a, "posthoc_pusher_error_difference": b})
    return result


def inputs():
    need(base.sha(RGB_SAVED) == RGB_SHA, "RGB source receipt SHA")
    saved = json.loads(RGB_SAVED.read_bytes())
    need(source.build() == saved and len(saved["rows"]) == 24, "fresh saved verification")
    _, data = source.physical.material()
    images = []
    for i, item in enumerate(saved["rows"]):
        need(item["index"] == i, "fixed ordered row")
        report = source.physical.native.read(source.DIR/f"row_{i:02d}/report.json", cap=64*1024**2)
        rec = [r for r in report["row"]["trajectory"] if r["phase"] == "probe"
               and r["step"] == item["returned_suffix"]]
        need(len(rec) == 1 and rec[0]["file_sha256"] == item["frame_sha256"], "fresh image receipt")
        path = source.DIR/f"row_{i:02d}"/rec[0]["png"]
        need(base.sha(path) == item["frame_sha256"], "frame file SHA")
        with Image.open(path) as f:
            need(f.format == "PNG" and f.mode == "RGB", "image format")
            current = np.asarray(f).copy()
        goal = np.asarray(data["roots"][item["identity"]["root_slot"]]["packets"]
                          [item["identity"]["condition"]]["goal_rgb"]).copy()
        need(current.shape == goal.shape == (224, 224, 3) and current.dtype == goal.dtype == np.uint8,
             "RGB ABI")
        images.append((current, goal))
    return saved, images


def run(output):
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT/"results" and output.name.startswith(NAME+"_")
         and not os.path.lexists(output), "exclusive output")
    saved, images = inputs()
    sources = base.bindings()
    for path in (Path(__file__), Path(source.__file__), Path(source.rgb.__file__), RGB_SAVED):
        sources[str(path.resolve())] = base.sha(path)
    with base.locking.gpu_lock(base.LOCK):
        output.mkdir(mode=0o700)
        manifest_sha = base.write(output/"manifest.json", {"schema": NAME+".manifest", "rows": 24,
                  "rgb_saved_sha256": RGB_SHA, "sources": sources, "model": base.MODEL_HASH,
                  "encoder_cap": CAP, "projector_cap": CAP, "wall_seconds": WALL,
                  "input_policy": "current and goal RGB only, all 24 rows including early stops",
                  "os_isolation_certified": False, "dependency_closure_complete": False,
                  "cooperative_lock_not_global": True, "P1_admission": False})
        counts, handles, values, error = {"encoder": 0, "projector": 0}, [], [], None
        host, before, after = None, None, None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from . import preflight_pusht_d2p_p0 as p0
            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
            before = base.model_hash(model)
            need(before == base.MODEL_HASH, "frozen model")
            classes = {str(Path(inspect.getfile(type(m))).resolve()) for m in model.modules()}
            base.write(output/"model_classes.json", {p: base.sha(p) for p in sorted(classes)})
            for key in counts:
                def count(module, args, key=key):
                    need(time.monotonic()-started < WALL and counts[key] < CAP
                         and torch.cuda.memory_reserved(device) < 6*1024**3, "resource cap")
                    counts[key] += 1
                handles.append(getattr(model, key).register_forward_pre_hook(count))
            transform = p0.image_transform()
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                for i, (current, goal) in enumerate(images):
                    latents = []
                    for img in (current, goal):
                        pixels = p0._image_tensor(img, transform=transform, device=device)
                        emb = model.encode({"pixels": pixels})["emb"]
                        need(emb.shape == (1, 1, 192) and bool(torch.isfinite(emb).all()), "encoding ABI")
                        latents.append(emb[0, 0])
                    d = float(torch.linalg.vector_norm(latents[0]-latents[1]).item())
                    values.append({"index": i, "latent_goal_distance": d,
                                   "current_pixel_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                                   "goal_pixel_sha256": hashlib.sha256(goal.tobytes()).hexdigest()})
            torch.cuda.synchronize(device)
            after = base.model_hash(model)
            need(before == after and counts == {"encoder": CAP, "projector": CAP}
                 and all(base.sha(path) == digest for path, digest in sources.items()), "model/source/call identity")
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        finally:
            for handle in handles:
                handle.remove()
        comparisons, rows = [], []
        if error is None:
            for val, item in zip(values, saved["rows"], strict=True):
                rows.append({**val, "identity": item["identity"], "full_horizon": item["full_horizon"],
                             "posthoc_pusher_error": item["posthoc_physical_metrics"]["pusher_position_error"],
                             "rgb_pusher_distance": (None if item["rgb_features"] is None else
                                 item["rgb_features"]["pusher_centroid_distance_pixels"])})
            comparisons = compare_order(rows)
        result = {"schema": NAME+".report", "status": "complete" if error is None and len(rows) == 24 else "failed",
                  "failure": error, "manifest_sha256": manifest_sha, "host": host, "model_before": before,
                  "model_after": after, "module_calls": counts, "rows": rows, "comparison": comparisons,
                  "comparison_counts": {kind: sum(x["status"] == kind for x in comparisons)
                                        for kind in ("agree", "disagree", "tie", "early_stop")},
                  "elapsed_seconds": time.monotonic()-started, "known_roots": True,
                  "native_truth_usage": "posthoc only; planner was not run", "physical_execution": False,
                  "os_isolation_certified": False, "independent_confirmation": False, "P1_admission": False}
        digest = base.write(output/"report.json", result)
        print(json.dumps({"status": result["status"], "sha256": digest, "failure": error,
                          "calls": counts, "comparison_counts": result["comparison_counts"]}), flush=True)
        return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))
