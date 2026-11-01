"""Optimize four equal-start second packets after real checkpoint-10 midpoint."""
import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_adam_budget_midpoint_cuda_20261031 as midpoint

ROOT = midpoint.ROOT
NAME = "p1_split_budget_feedback_plan_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_SPLIT_BUDGET_FEEDBACK_PROTOCOL_20261031.md"
need = midpoint.need


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    terminal = midpoint.terminal
    terminal.curve.planner.configure()
    base = terminal.curve.planner.adam.prior.base
    prior = terminal.curve.planner.adam.prior
    all_entries, stopped, sources = midpoint.inputs()
    entries = [x for x in all_entries if x[0]["step"] == 10]
    need(len(entries) == 24 and not any(x["step"] == 10 for x in stopped)
         and [x[0]["index"] for x in entries] == list(range(24)), "fixed complete 10-update roster")
    for path, digest in base.bindings().items():
        need(path not in sources or sources[path] == digest, "source collision")
        sources[path] = digest
    for path in (Path(__file__).resolve(), PROTOCOL, Path(midpoint.feedback.__file__).resolve()):
        sources[str(path)] = base.sha(path)
    rows = []
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        need(before == base.MODEL_HASH, "frozen checkpoint")
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type="cuda", enabled=False):
            for entry, _, pixels, pixel_sha in entries:
                row, raw = entry["row"], entry["action"]
                with Image.open(io.BytesIO(base64.b64decode(row["cp5_png_base64"], validate=True))) as im:
                    cp_rgb = np.asarray(im.convert("RGB")).copy()
                with torch.no_grad():
                    def encode(im):
                        return model.encode({"pixels": p0._image_tensor(
                            im, transform=transform, device=device)})["emb"][0, 0]
                    cp, mid = encode(cp_rgb), encode(pixels)
                    states, history, goal, mean, std = midpoint.feedback.second_context(row, cp, mid, raw[0])
                starts = np.asarray(prior.initial_actions(row["logical_pair"])[:, 1], dtype="<f4")
                need(starts.shape == (4, 5, 2), "starts")
                candidates, scores = [], []
                torch.cuda.synchronize(device)
                start = time.monotonic()
                for initial in starts:
                    action = torch.nn.Parameter(torch.tensor(initial[None], dtype=torch.float32, device=device))
                    optim = torch.optim.Adam([action], lr=0.05)

                    def score():
                        token = ((action - mean) / std).reshape(1, 1, 10)
                        prediction = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))
                        need(prediction.shape == states.shape and bool(torch.isfinite(prediction).all()), "prediction")
                        return (prediction[:, -1] - goal).square().mean()

                    for _ in range(10):
                        optim.zero_grad(set_to_none=True)
                        score().backward()
                        need(action.grad is not None and bool(torch.isfinite(action.grad).all()), "gradient")
                        optim.step()
                        with torch.no_grad():
                            action.clamp_(-1, 1)
                    with torch.no_grad():
                        scores.append(float(score()))
                    candidates.append(action.detach()[0].cpu().numpy().copy())
                need(np.isfinite(scores).all(), "scores")
                torch.cuda.synchronize(device)
                winner = prior.pick(scores)
                chosen = np.asarray(candidates[winner], dtype="<f4")
                rows.append({"index": entry["index"], "logical_pair": row["logical_pair"], "role": row["role"],
                             "first_packet_hex": raw[0].tobytes().hex(), "second_packet_hex": chosen.tobytes().hex(),
                             "midpoint_pixel_sha256": pixel_sha,
                             "initial_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
                             "scores": scores, "winner": winner, "planning_wall_seconds": time.monotonic() - start})
                print(json.dumps({"index": entry["index"], "winner": winner}), flush=True)
        torch.cuda.synchronize(device)
        after = base.model_hash(model)
        need(before == after and all(base.sha(p) == h for p, h in sources.items()), "source or model drift")
        result = {"schema": NAME, "status": "complete", "rows": rows, "sources": sources, "host": host,
                  "model_before": before, "model_after": after,
                  "per_row_calls_code_path": {"encoder": 2, "predict": 44, "action_encoder": 44,
                                              "backward": 40},
                  "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
                  "limitations": "Known roots; retrospective checkpoint-10; first-stage historical run "
                                 "incurred extra work; offline midpoint; no total-budget certification."}
        digest = base.write(output, result)
    print(json.dumps({"rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print("Optimize all 24 fixed checkpoint-10 midpoints with four equal starts on host CUDA.", flush=True)
    run(parser.parse_args().output)
