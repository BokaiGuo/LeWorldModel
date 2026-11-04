"""Isolated prediction of a fixed evaluator-supplied action bank; no planning."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pc_jepa_v7_isolated_worker_20261104_codex import ROOT, LOCK, UUID, CAP, CHUNK, sha
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability import pc_response_temporal_search_v1 as temporal
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_recursive_response_v12 as v12
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history


def main():
    base = Path(sys.argv[1])
    inp = base / "input"
    config = json.loads((inp / "config.json").read_text())
    step, arm = config["step"], config["arm"]
    if step not in (0, 1, 2) or arm not in ("response", "cem"):
        raise ValueError("invalid decision")
    for p in (ROOT / "data/jepa_wm_osf", ROOT / "results/p5_jepa_factual_nonzero_20261027_v2",
              ROOT / "results/pc_jepa_native_boundary_replay_20261104_codex"):
        if p.exists():
            raise ValueError(f"forbidden source visible: {p}")
    metadata = json.loads((inp / "model_metadata.json").read_text())
    with np.load(inp / "context.npz", allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    shapes = {"rgb_history": (3, 224, 224, 3), "goal_rgb": (1, 224, 224, 3),
              "normalized_proprio": (3, 4), "normalized_history_packets": (2, 10),
              "action_mean": (2,), "action_std": (2,), "carry": (15 - 5 * step, 2) if step else (0, 2)}
    if set(data) != set(shapes) or any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError("invalid planner input")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("CUDA device 0 required")
    if str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") != UUID:
        raise ValueError("unexpected GPU")
    with gpu_lock(LOCK):
        models, strict, sources, before = native.load_cuda(ROOT, metadata)
        _, v5, _ = native.helper(ROOT).dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import SDPBackend, sdpa_kernel
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            start = time.monotonic()
            def encode(images):
                pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                return models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            context = encode(data["rgb_history"])
            goal = encode(data["goal_rgb"]).reshape(-1)
            visual = context.reshape(1, 3, 1, 16, 16, 384)
            features = models["proprio_encoder"](torch.from_numpy(data["normalized_proprio"]).unsqueeze(0).to("cuda")).expand(1, 3, 256, 16)
            history = torch.from_numpy(data["normalized_history_packets"]).unsqueeze(0).to("cuda")
            mean, std = (torch.as_tensor(data[k], device="cuda") for k in ("action_mean", "action_std"))
            if torch.any(std <= 0):
                raise ValueError("invalid std")
            root = context[-1].reshape(-1)
            delta = goal - root
            norm = torch.linalg.vector_norm(delta)
            if norm.item() <= 1e-8:
                raise ValueError("degenerate chart")
            axis = delta / norm
            random = torch.as_tensor(np.random.default_rng(1).normal(size=axis.numel()).astype(np.float32), device="cuda")
            random = random - torch.dot(random, axis) * axis
            chart = torch.stack((axis, random / torch.linalg.vector_norm(random)), dim=1) / np.sqrt(axis.numel())
            target = (delta @ chart).cpu().numpy()
            torch.cuda.synchronize()
            encoding_seconds = time.monotonic() - start
            core, meter, records = NativeCore(models["predictor"]), Meter(CAP), []

            def evaluate(actions):
                a = np.asarray(actions, dtype=np.float32)
                if a.ndim != 3 or a.shape[1] not in (5, 10, 15) or a.shape[2] != 2 or not np.isfinite(a).all() or np.abs(a).max() > 1.000001:
                    raise ValueError("invalid action")
                a = np.clip(a, -1, 1)
                def model_eval(sample):
                    positions, scores = [], []
                    for off in range(0, len(sample), CHUNK):
                        chunk = torch.as_tensor(sample[off:off + CHUNK], device="cuda")
                        packet = ((chunk - mean) / std).reshape(len(chunk), -1, 10).transpose(0, 1)
                        pred = unroll_history(core, visual, features, history, packet, 3)["visual"][-1]
                        flat = pred.reshape(len(chunk), -1)
                        positions.append(((flat - root) @ chart).cpu().numpy())
                        scores.append((flat - goal).square().mean(dim=1).cpu().numpy())
                    torch.cuda.synchronize()
                    return np.concatenate(positions), np.concatenate(scores)
                pos, score = meter.predict(a, model_eval)
                records.append({"actions": a.tolist(), "positions": pos.tolist(), "scores": score.tolist()})
                return pos, score

            start = time.monotonic()
            horizon = 15 - 5 * step
            carry = data["carry"] if step else None
            if step != 0 or arm != "response":
                raise ValueError("only sealed initial response context is supported")
            bank = np.load(inp / "bank.npy", allow_pickle=False)
            if bank.dtype != np.float32 or bank.shape != (64, 15, 2):
                raise ValueError("invalid evaluation bank")
            position, scores = evaluate(bank)
            decision = {"scope": "evaluation-only; no selection",
                        "scores": scores.tolist(), "positions": position.tolist()}
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            if before != {key: v4.model_receipt(model) for key, model in models.items()}:
                raise ValueError("model changed")
    output = {"arm": arm, "step": step, "input_sha256": sha(inp / "context.npz"),
              "bank_sha256": sha(inp / "bank.npy"),
              "config_sha256": sha(inp / "config.json"), "metadata_sha256": sha(inp / "model_metadata.json"),
              "worker_sha256": sha(Path(__file__)), "decision": decision, "evaluations": records,
              "meter": meter.as_dict(), "predictor_module_calls": core.calls, "chart_goal": target.tolist(),
              "encoding_seconds": encoding_seconds, "planning_seconds": elapsed,
              "strict_load": strict, "imported_sources": sources}
    with (base / "output/decision.json").open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"fixed bank: token={meter.token_samples} samples={len(decision['scores'])}", flush=True)


if __name__ == "__main__":
    main()
