"""RGB-only planner: no import or read of evaluation datasets or outcomes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_rgb_derivative_backend_20261107_codex import precise_action_derivatives
from experiments.pc_rgb_goal_gradient_candidates_20261107_codex import make_candidates
from experiments.pc_rgb_single_frame_contract_20261107_codex import SingleFrameRGB, model_digest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/pc_rgb_gradient_public_input_20261107_codex.npz"
OUT = ROOT / "results/pc_rgb_gradient_public_plan_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_RGB_GRADIENT_PROCESS_SEPARATION_PROTOCOL_20261107_CODEX.md"
CHECKPOINT = Path("/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
KEYS = {"episodes", "current", "goal", "nominal", "action_mean", "action_scale"}


def normalize_actions(raw, mean, scale):
    mean = torch.as_tensor(mean, dtype=raw.dtype, device=raw.device)
    scale = torch.as_tensor(scale, dtype=raw.dtype, device=raw.device)
    if raw.shape != (25, 2) or mean.shape != (2,) or scale.shape != (2,) or (scale <= 0).any():
        raise ValueError("invalid action or normalizer")
    return ((raw - mean) / scale).reshape(1, 1, 5, 10)


def validate_public(data):
    if set(data.files) != KEYS:
        raise ValueError("unexpected public input fields")
    episodes, current, goal, nominal = (data[key] for key in ("episodes", "current", "goal", "nominal"))
    if (episodes.ndim != 1 or len(episodes) == 0 or len(set(episodes.tolist())) != len(episodes)
            or current.shape != goal.shape or current.ndim != 4 or current.shape[0] != len(episodes)
            or current.shape[-1] != 3 or current.dtype != np.uint8 or goal.dtype != np.uint8
            or nominal.shape != (len(episodes), 25, 2) or not np.isfinite(nominal).all()
            or np.abs(nominal).max() > .7000001):
        raise ValueError("invalid public data")
    mean, scale = data["action_mean"], data["action_scale"]
    if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid public normalizer")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    model = load_compatible_pretrained(str(CHECKPOINT), device="cuda")
    digest = model_digest(model)
    audit = freeze_audit(RGB, "lewm", [__file__, PUBLIC, PROTOCOL, CHECKPOINT / "weights.pt",
                         CHECKPOINT / "config.json"], preprocessing="ImageNet 224; fixed action statistics",
                         clock="five environment steps per packet")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    rows = []
    print("Plan using only public RGB and nominal actions; no evaluation data.", flush=True)
    with np.load(PUBLIC, allow_pickle=False) as data:
        validate_public(data)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.n_features_in_ = data["action_mean"], data["action_scale"], 2
        guard = SingleFrameRGB(model, audit, expected_model_digest=digest, action_mean=scaler.mean_,
                               action_scale=scaler.scale_, packet_steps=5, physical_low=-1, physical_high=1)
        for i, episode in enumerate(data["episodes"]):
            context = {"pixels": transform(data["current"][i])[None, None, None].cuda(),
                       "goal": transform(data["goal"][i])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            raw = torch.tensor(data["nominal"][i], device="cuda", requires_grad=True)
            with precise_action_derivatives():
                cost, _ = guard.score(context, normalize_actions(raw, scaler.mean_, scaler.scale_))
                gradient, = torch.autograd.grad(cost[0, 0], raw)
            actions, direction = make_candidates(data["nominal"][i], gradient.detach().cpu().numpy())
            norm = scaler.transform(actions.reshape(-1, 2)).astype(np.float32).reshape(1, 5, 5, 10)
            with torch.no_grad():
                scores, _ = guard.score(context, torch.tensor(norm, device="cuda"))
            values = scores[0].cpu().numpy()
            rows.append({"episode": int(episode), "candidates": actions.tolist(), "direction": direction.tolist(),
                         "gradient": gradient.detach().cpu().tolist(), "scores": values.tolist(),
                         "selected_index": int(np.argmin(values))})
            print(f"episode={episode} selected={rows[-1]['selected_index']}", flush=True)
        guard.verify_unchanged()
    with OUT.open("x") as stream:
        json.dump({"track": RGB, "scope": "public input planner, exposed development",
                   "public_sha256": file_hash(PUBLIC), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(__file__), "model_digest": digest, "rows": rows},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
