"""Plan a fixed ten-step suffix from each available real cp5 on host CUDA."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_multistart_cp5_20261030_main as common
from . import p1_hybrid_cp5_bundle_20261031 as source

NAME = "p1_hybrid_cp5_adam_20261031"
base = common.base


def run(bundle_path, bundle_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    need = source.need
    need(base.sha(bundle_path) == bundle_sha, "bundle SHA")
    bundle = json.loads(Path(bundle_path).read_bytes())
    need(bundle == source.build(), "reconstructed bundle")
    need(bundle["schema"] == source.NAME and bundle["fixed_rows"] == len(bundle["rows"]) == 2
         and bundle["P1_admission"] is False and bundle["independent_confirmation"] is False,
         "fixed bundle")
    for i, item in enumerate(bundle["rows"]):
        need(item["index"] == i and item["logical_pair"] in (0, 4), "fixed row order")
        source.blind.validate_row(item["blind"])
        need(item["blind"]["logical_pair"] == item["logical_pair"], "initialization pair")
    output = Path(output)
    need(output.is_absolute() and output.parent == base.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive output")
    sources = base.bindings()
    sources.update(bundle["bindings"])
    for path in (Path(__file__).resolve(), Path(common.__file__).resolve(), source.PROTOCOL, Path(bundle_path)):
        sources[str(path)] = base.sha(path)
    need(all(base.sha(p) == d for p, d in sources.items()), "source binding")
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen checkpoint")
        transform, rows, start = p0.image_transform(), [], time.monotonic()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for item in bundle["rows"]:
                row = item["blind"]
                identity = {k: item[k] for k in ("index", "logical_pair", "root_slot")}
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as image:
                    pixels = np.asarray(image).copy()
                with torch.no_grad():
                    cp5 = model.encode({"pixels": p0._image_tensor(pixels, transform=transform,
                                                                   device=device)})["emb"][0, 0]
                initial = common.initial_actions(item["logical_pair"])
                candidates, scores = [], []
                row_started = time.monotonic()
                for init in initial:
                    action = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
                    optimizer = torch.optim.Adam([action], lr=0.05)
                    for _ in range(20):
                        optimizer.zero_grad(set_to_none=True)
                        pred, goal = common.fixed.replay(model, row, cp5, action)
                        loss = (pred - goal).square().mean()
                        loss.backward()
                        need(action.grad is not None and bool(torch.isfinite(action.grad).all()), "gradient")
                        optimizer.step()
                        with torch.no_grad():
                            action.clamp_(-1, 1)
                    with torch.no_grad():
                        pred, goal = common.fixed.replay(model, row, cp5, action)
                        scores.append(float((pred - goal).square().mean()))
                    candidates.append(action.detach()[0].cpu().numpy().copy())
                torch.cuda.synchronize(device)
                winner = common.pick(scores)
                selected = np.asarray(candidates[winner], dtype="<f4")
                need(np.isfinite(selected).all() and (np.abs(selected) <= 1).all(), "finite actions")
                rows.append({**identity, "status": "planned", "cp5_pixel_sha256": row["cp5_pixel_sha256"],
                             "prefix_raw_f32_hex": row["prefix_raw_f32_hex"],
                             "initial_f32_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                             "candidate_scores": scores, "selected_index": winner,
                             "actions": selected.tolist(), "actions_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                             "wall_seconds": time.monotonic() - row_started,
                             "calls": {"action_encoder": 168, "predict": 168, "backward": 80,
                                       "post_update_candidate_scores": 4, "encoder": 1}})
                print(json.dumps({"index": item["index"], "status": "planned"}), flush=True)
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(after == before and all(base.sha(p) == d for p, d in sources.items()), "model/source drift")
        count = sum(r["status"] == "planned" for r in rows)
        result = {"schema": NAME, "status": "complete", "rows": rows, "fixed_rows": 2,
                  "planned_rows": count, "host": host, "sources": sources, "model_before": before,
                  "model_after": after, "elapsed_excluding_load": time.monotonic() - start,
                  "config": {"starts": 4, "steps_per_start": 20, "optimizer": "Adam",
                             "learning_rate": 0.05, "horizon": 10, "action_box": [-1, 1]},
                  "calls": {"action_encoder": 168 * count, "predict": 168 * count,
                            "backward": 80 * count, "encoder": count},
                  "P1_admission": False, "independent_confirmation": False, "physical_execution": False,
                  "limitations": "Known dependent roots, code-derived counts not independent hooks; "
                                 "prefix cost not included, no total equal-budget certification."}
        digest = base.write(output, result)
    print(json.dumps({"status": "complete", "planned": count, "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Plan fixed real-cp5 suffixes using the same four starts and optimizer on CUDA.", flush=True)
    run(args.bundle, args.bundle_sha256, args.output)
