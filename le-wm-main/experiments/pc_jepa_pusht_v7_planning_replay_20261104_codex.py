"""Replay every saved candidate with the same JEPA checkpoint, no optimization."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_pusht_v7_planning_dev_20261104_codex.json"
OUTPUT = ROOT / "results/pc_jepa_pusht_v7_planning_replay_20261104_codex.json"
MANIFEST = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
EXPECTED = "10a8cfa21c7821cd97b69cfd2c1078c9a0534c706f8251f1276dc9f88a154466"
UUID = "4e8973ff-0278-53f3-2bae-2615d461e441"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists() or sha(SOURCE) != EXPECTED:
        raise ValueError("output exists or input hash differs")
    source = json.loads(SOURCE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    native.verify_manifest(ROOT, manifest)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("physical GPU required")
    if str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") != UUID:
        raise ValueError("GPU UUID mismatch")
    with gpu_lock(LOCK):
        models, _, _, before = native.load_cuda(ROOT, manifest)
        _, v5, _ = native.helper(ROOT).dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        arrays = v5.read_arrays(v4, ROOT)
        stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
        values, rgb, _ = v5.row_inputs(v4, ROOT, arrays, stats, native.EPISODE)
        if v4.array_sha(rgb) != source["history_rgb_sha256"]:
            raise ValueError("history mismatch")
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import decord

        video_path = v4.data_root(ROOT) / "obses/episode_001.mp4"
        if sha(video_path) != source["video_sha256"]:
            raise ValueError("video hash mismatch")
        reader = decord.VideoReader(str(video_path), num_threads=1, ctx=decord.cpu(0))
        goal_rgb = reader.get_batch([25]).asnumpy()
        if v4.array_sha(goal_rgb) != source["goal_rgb_sha256"]:
            raise ValueError("goal mismatch")
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            def encode(images):
                pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                return models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            patches = encode(rgb[:3])
            goal = encode(goal_rgb).reshape(-1)
            visual = patches.reshape(1, 3, 1, 16, 16, 384)
            prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            history = torch.from_numpy(values["packets"][:2].copy()).unsqueeze(0).to("cuda")
            mean = torch.tensor(stats["ACTION_MEAN"], dtype=torch.float32, device="cuda")
            std = torch.tensor(stats["ACTION_STD"], dtype=torch.float32, device="cuda")
            root = patches[-1].reshape(-1)
            delta = goal-root
            axis0 = delta / torch.linalg.vector_norm(delta)
            rand = torch.as_tensor(np.random.default_rng(1).normal(size=axis0.numel()).astype(np.float32), device="cuda")
            rand = rand-torch.dot(rand, axis0)*axis0
            chart = torch.stack((axis0, rand / torch.linalg.vector_norm(rand)), dim=1) / np.sqrt(axis0.numel())
            core, reports = NativeCore(models["predictor"]), {}
            for arm, row in source["results"].items():
                calls_before = core.calls
                max_score = max_position = 0.
                samples = tokens = 0
                for event in row["evaluations"]:
                    actions = np.asarray(event["actions"], dtype=np.float32)
                    positions, scores = [], []
                    for offset in range(0, len(actions), 4):
                        chunk = torch.as_tensor(actions[offset:offset+4], device="cuda")
                        packets = ((chunk-mean)/std).reshape(len(chunk), -1, 10).transpose(0, 1)
                        predicted = unroll_history(core, visual, features, history, packets, 3)["visual"][-1]
                        flat = predicted.reshape(len(chunk), -1)
                        positions.append(((flat-root) @ chart).cpu().numpy())
                        scores.append((flat-goal).square().mean(dim=1).cpu().numpy())
                    p, s = np.concatenate(positions), np.concatenate(scores)
                    max_position = max(max_position, float(np.max(np.abs(p-np.asarray(event["positions"])))))
                    max_score = max(max_score, float(np.max(np.abs(s-np.asarray(event["scores"])))))
                    samples += len(actions)
                    tokens += len(actions)*actions.shape[1]//5
                reports[arm] = {"calls": len(row["evaluations"]), "samples": samples,
                    "token_samples": tokens, "predictor_calls": core.calls-calls_before,
                    "position_max_abs_error": max_position, "score_max_abs_error": max_score}
                if samples != row["meter"]["trajectory_samples"] or tokens != row["meter"]["prediction_token_samples"]:
                    raise ValueError("sample count mismatch")
                if max_position > 1e-5 or max_score > 1e-5:
                    raise ValueError("saved predictions do not replay")
                print(f"{arm}: replayed {samples} trajectories", flush=True)
            torch.cuda.synchronize()
            if before != {key: v4.model_receipt(model) for key, model in models.items()}:
                raise ValueError("model changed")
    output = {"scope": "same-checkpoint deterministic request replay; not independent model or physical replay",
              "source_sha256": EXPECTED, "script_sha256": sha(Path(__file__)), "arms": reports}
    with OUTPUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
