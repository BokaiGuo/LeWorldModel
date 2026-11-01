"""Live four-start first packet followed by live-RGB trust replanning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_live_best_20261030_main as live
from . import codex_p1_adam_cp5_20261030_main as adam

launcher = live.launcher
ROOT = launcher.ROOT
NAME = "codex_p1_live_twoplan_20261030_main"
ORIGINAL_BUILD = launcher.build


def build():
    result = ORIGINAL_BUILD()
    sources = dict(result["sources"])
    for module in (live, adam, adam.prior, adam.prior.fixed, live.best, live.source):
        path = Path(module.__file__).resolve()
        sources[str(path)] = launcher.sha(path)
    path = Path(__file__).resolve()
    sources[str(path)] = launcher.sha(path)
    result.update(schema=NAME+".prepare", sources=sources,
                  method="live_4start_adam_then_live_rgb_trust_best_iterate",
                  explicit_path_counts={"stage1_encoder": 1, "stage1_predict": 168, "stage1_backward": 80,
                                        "stage2_encoder": 1, "stage2_predict": 22, "stage2_backward": 20},
                  optimizer={"stage1_starts": 4, "stage1_steps": 20, "stage2_steps": 20,
                             "learning_rate": 0.05, "stage2_trust_weight": 0.1},
                  limitations="known fixed prefixes, explicit code path counts, no full hook or fair total budget")
    return result


def run_stage(folder, row, stage, png, previous_latent, executed_packet,
              model, transform, device, source_hashes):
    import torch
    from . import preflight_pusht_d2p_p0 as p0

    if stage == 2:
        return live.run_stage(folder, row, stage, png, previous_latent, executed_packet,
                              model, transform, device, source_hashes)
    if stage != 1 or previous_latent is not None or executed_packet is not None:
        raise ValueError("invalid first-stage input")
    pixels = live.source.feedback.rgb_pixels(png)
    pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    if pixel_hash != row["cp5_pixel_sha256"]:
        raise ValueError("cp5 image identity")
    folder = Path(folder)
    if not folder.is_absolute() or folder.exists() or folder.is_symlink() or not folder.parent.is_dir():
        raise ValueError("exclusive stage")
    folder.mkdir(mode=0o700)
    with torch.no_grad():
        latent = model.encode({"pixels": p0._image_tensor(pixels, transform=transform, device=device)})["emb"][0, 0]
    initial = adam.prior.initial_actions(row["logical_pair"])
    candidates, scores = [], []
    for init in initial:
        action = torch.nn.Parameter(torch.tensor(init[None], dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([action], lr=adam.LR)
        for _ in range(adam.STEPS):
            optimizer.zero_grad(set_to_none=True)
            pred, goal = adam.prior.fixed.replay(model, row, latent, action)
            loss = (pred-goal).square().mean()
            loss.backward()
            if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            with torch.no_grad():
                action.clamp_(-1, 1)
        with torch.no_grad():
            pred, goal = adam.prior.fixed.replay(model, row, latent, action)
            scores.append(float((pred-goal).square().mean()))
        candidates.append(action.detach()[0].cpu().numpy().copy())
    if len(scores) != 4 or not np.isfinite(scores).all():
        raise ValueError("candidate scores")
    winner = adam.prior.pick(scores)
    raw = np.asarray(candidates[winner], dtype="<f4")
    if raw.shape != (2, 5, 2):
        raise ValueError("horizon shape")
    live.source.feedback.packet_bytes(raw[0])
    live.source.feedback.packet_bytes(raw[1])
    # Only the selected model-generated plan becomes the next-stage reference.
    live.ENTRIES[row["index"]] = (row, None, raw, None, None)
    result = {"raw_horizon_f32_hex": raw.tobytes().hex(), "execute_packet": 0,
              "latent_sha256": launcher.stage.tensor_hash(latent)}
    record = {"index": row["index"], "stage": 1, "status": "complete", "pixel_sha256": pixel_hash,
              "png_sha256": hashlib.sha256(png).hexdigest(), "source_hashes": source_hashes,
              "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
              "candidate_scores": scores, "selected_index": winner, "result": result,
              "explicit_path_counts": {"encoder": 1, "predict": 168, "backward": 80}}
    launcher.base.write(folder/"report.json", record)
    return latent, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    args = ap.parse_args()
    launcher.NAME, launcher.build, launcher.stage.run_stage = NAME, build, run_stage
    if args.command == "prepare":
        print(json.dumps({"sha256": launcher.prepare(args.output)}))
        return 0
    if not args.prepared or not args.sha256:
        raise ValueError("prepared contract required")
    live.ENTRIES = [None] * 24
    return launcher.run(args.prepared, args.sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
