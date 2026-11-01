"""Fixed-first-packet live-RGB best-iterate replanning on the known P1 panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_two_stage_launcher_20261030_v1 as launcher
from . import codex_p1_adam_feedback_plan_20261030_main as source
from . import codex_p1_adam_best_iterate_20261030_main as best

NAME = "codex_p1_live_best_20261030_main"
ORIGINAL_BUILD = launcher.build
ROOT = launcher.ROOT
ENTRIES = None


def build():
    result = ORIGINAL_BUILD()
    sources = dict(result["sources"])
    for module in (source, best):
        path = Path(module.__file__).resolve()
        sources[str(path)] = launcher.sha(path)
    path = Path(__file__).resolve()
    sources[str(path)] = launcher.sha(path)
    result.update(schema=NAME+".prepare", sources=sources,
                  method="fixed_adam_first_packet_then_live_rgb_trust_best_iterate",
                  explicit_path_counts={"stage1_encoder": 1, "stage2_encoder": 1,
                                        "stage2_predict": 22, "stage2_backward": 20},
                  optimizer={"steps": 20, "learning_rate": 0.05, "trust_weight": 0.1},
                  limitations="fixed first packet, known roots, no complete model hook or equal-budget proof")
    return result


def run_stage(folder, row, stage, png, previous_latent, executed_packet,
              model, transform, device, source_hashes):
    import torch
    from . import preflight_pusht_d2p_p0 as p0

    entry = ENTRIES[row["index"]]
    original, _, raw, _, _ = entry
    if original["index"] != row["index"] or original["role"] != row["role"]:
        raise ValueError("fixed input identity")
    pixels = source.feedback.rgb_pixels(png)
    pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    folder = Path(folder)
    if not folder.is_absolute() or folder.exists() or folder.is_symlink() or not folder.parent.is_dir():
        raise ValueError("exclusive stage")
    folder.mkdir(mode=0o700)
    def encode(img):
        return model.encode({"pixels": p0._image_tensor(img, transform=transform, device=device)})["emb"][0, 0]
    with torch.no_grad():
        latent = encode(pixels)
    if stage == 1:
        if previous_latent is not None or executed_packet is not None or pixel_hash != row["cp5_pixel_sha256"]:
            raise ValueError("cp5 input")
        packet = raw[0].copy()
        scores, chosen = [], None
    elif stage == 2:
        if previous_latent is None or not np.array_equal(executed_packet, raw[0]):
            raise ValueError("executed first packet")
        with torch.no_grad():
            states, history, goal, mean, std = source.feedback.second_context(row, previous_latent, latent, executed_packet)
        ref = torch.tensor(raw[1][None], dtype=torch.float32, device=device)
        action = torch.nn.Parameter(ref.clone())
        optimizer = torch.optim.Adam([action], lr=0.05)
        def score():
            token = ((action-mean)/std).reshape(1, 1, 10)
            pred = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))[:, -1]
            return (pred-goal).square().mean() + 0.1*(action-ref).square().mean()
        with torch.no_grad():
            baseline = float(score())
        scores, packets = [], []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            value = score()
            scores.append(float(value.detach()))
            packets.append(action.detach().clone())
            value.backward()
            if action.grad is None or not bool(torch.isfinite(action.grad).all()):
                raise ValueError("gradient")
            optimizer.step()
            with torch.no_grad():
                action.clamp_(-1, 1)
        with torch.no_grad():
            scores.append(float(score()))
            packets.append(action.detach().clone())
        chosen = best.select_best(scores)
        if abs(scores[0]-baseline) > 1e-6:
            raise ValueError("baseline mismatch")
        packet = packets[chosen].cpu().numpy()[0].astype("<f4")
    else:
        raise ValueError("stage")
    source.feedback.packet_bytes(packet)
    raw_horizon = np.concatenate((packet, np.zeros((5, 2), dtype="<f4")), axis=0).astype("<f4")
    result = {"raw_horizon_f32_hex": raw_horizon.tobytes().hex(), "execute_packet": 0,
              "latent_sha256": launcher.stage.tensor_hash(latent)}
    record = {"index": row["index"], "stage": stage, "status": "complete", "pixel_sha256": pixel_hash,
              "png_sha256": hashlib.sha256(png).hexdigest(), "selected_step": chosen, "scores": scores,
              "executed_history_sha256": None if executed_packet is None else hashlib.sha256(executed_packet.tobytes()).hexdigest(),
              "result": result, "source_hashes": source_hashes,
              "explicit_path_counts": {"encoder": 1, "predict": 22 if stage == 2 else 0,
                                       "backward": 20 if stage == 2 else 0}}
    launcher.base.write(folder/"report.json", record)
    return latent, result


def main():
    global ENTRIES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    args = ap.parse_args()
    launcher.NAME, launcher.build = NAME, build
    launcher.stage.run_stage = run_stage
    if args.command == "prepare":
        print(json.dumps({"sha256": launcher.prepare(args.output)}))
        return 0
    if not args.prepared or not args.sha256:
        raise ValueError("prepared contract required")
    ENTRIES = source.inputs()
    return launcher.run(args.prepared, args.sha256, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
