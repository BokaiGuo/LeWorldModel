"""Exposed JEPA PushT model-only v7/CEM integration, not physical evaluation."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability import pc_response_temporal_search_v1 as temporal
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_JEPA_PUSHT_V7_PLANNING_DEV_PROTOCOL_20261104_CODEX.md"
MANIFEST = ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json"
OUTPUT = ROOT / "results/pc_jepa_pusht_v7_planning_dev_20261104_codex.json"
FAILURE = ROOT / "results/pc_jepa_pusht_v7_planning_dev_20261104_codex_failure.json"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
UUID = "4e8973ff-0278-53f3-2bae-2615d461e441"
CAP, CHUNK = 1920, 4


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_exclusive(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")


def run():
    if OUTPUT.exists() or FAILURE.exists():
        raise FileExistsError("existing result or failure; no retry on this protocol")
    print("JEPA PushT v7/CEM: model-only, exposed goal, no physical execution", flush=True)
    manifest = json.loads(MANIFEST.read_text())
    native.verify_manifest(ROOT, manifest)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 required")
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    if uuid != UUID:
        raise RuntimeError("unexpected physical GPU")
    with gpu_lock(LOCK):
        models, strict, sources, before = native.load_cuda(ROOT, manifest)
        helper = native.helper(ROOT)
        _, v5, _ = helper.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        arrays = v5.read_arrays(v4, ROOT)
        stats = v4.source_stats(v4.native_root(ROOT) / "app/plan_common/datasets/pusht_dset.py")
        values, rgb, length = v5.row_inputs(v4, ROOT, arrays, stats, native.EPISODE)
        saved = manifest["row"]
        if v4.array_sha(rgb) != saved["rgb_sha256"] or any(
            v4.array_sha(value) != saved["inputs"][key]["sha256"] for key, value in values.items()
        ):
            raise ValueError("history binding mismatch")
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import decord

        video_path = v4.data_root(ROOT) / "obses/episode_001.mp4"
        reader = decord.VideoReader(str(video_path), num_threads=1, ctx=decord.cpu(0))
        goal_rgb = reader.get_batch([25]).asnumpy()
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            started = time.monotonic()
            def encode(images):
                pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                return models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            context_patches = encode(rgb[:3])
            goal = encode(goal_rgb).reshape(-1)
            visual = context_patches.reshape(1, 3, 1, 16, 16, 384)
            prop = torch.from_numpy(values["normalized_proprio"][:3].copy()).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            history = torch.from_numpy(values["packets"][:2].copy()).unsqueeze(0).to("cuda")
            mean = torch.tensor(stats["ACTION_MEAN"], dtype=torch.float32, device="cuda")
            std = torch.tensor(stats["ACTION_STD"], dtype=torch.float32, device="cuda")
            if torch.any(std <= 0):
                raise ValueError("invalid action statistics")
            root_latent = context_patches[-1].reshape(-1)
            delta = goal - root_latent
            norm = torch.linalg.vector_norm(delta)
            if norm.item() <= 1e-8:
                raise ValueError("degenerate goal chart")
            axis0 = delta / norm
            r = torch.as_tensor(np.random.default_rng(1).normal(size=axis0.numel()).astype(np.float32), device="cuda")
            r = r - torch.dot(r, axis0) * axis0
            axis1 = r / torch.linalg.vector_norm(r)
            chart = torch.stack((axis0, axis1), dim=1) / np.sqrt(axis0.numel())
            chart_goal = (delta @ chart).cpu().numpy()
            torch.cuda.synchronize()
            encode_seconds = time.monotonic() - started
            core = NativeCore(models["predictor"])
            results = {}
            for arm in ("v7_response", "smooth_cem"):
                meter, records = Meter(CAP), []
                before_calls = core.calls

                def evaluate(actions):
                    a = np.asarray(actions, dtype=np.float32)
                    if a.ndim != 3 or a.shape[1] not in (5, 10, 15) or a.shape[2] != 2:
                        raise ValueError("invalid trajectory shape")
                    if not np.isfinite(a).all() or np.abs(a).max() > 1.000001:
                        raise ValueError("raw action outside [-1,1]")
                    a = np.clip(a, -1, 1)
                    def model_eval(sample):
                        positions, scores = [], []
                        for offset in range(0, len(sample), CHUNK):
                            chunk = torch.as_tensor(sample[offset:offset+CHUNK], device="cuda")
                            packet = ((chunk - mean) / std).reshape(len(chunk), -1, 10).transpose(0, 1)
                            pred = unroll_history(core, visual, features, history, packet, 3)["visual"][-1]
                            flat = pred.reshape(len(chunk), -1)
                            positions.append(((flat-root_latent) @ chart).cpu().numpy())
                            scores.append((flat-goal).square().mean(dim=1).cpu().numpy())
                        torch.cuda.synchronize()
                        return np.concatenate(positions), np.concatenate(scores)
                    positions, scores = meter.predict(a, model_eval)
                    records.append({"actions": a.tolist(), "positions": positions.tolist(), "scores": scores.tolist()})
                    if len(records) % 30 == 0:
                        print(f"{arm}: calls={len(records)} tokens={meter.token_samples}/{CAP}", flush=True)
                    return positions, scores

                started = time.monotonic()
                if arm == "v7_response":
                    output = v7.decide("response", "response", 0, 0, chart_goal,
                                       evaluate, meter, temporal.NOMINALS, None)
                else:
                    initial = np.zeros((15, 2))
                    initial_score = float(evaluate(initial[None])[1][0])
                    count = (CAP - meter.token_samples) // 3
                    chain, score, generations = v6.cem(initial, [], count,
                        np.random.default_rng(2027110401), evaluate, initial_score, 4)
                    output = {"selected": {"name": "smooth_cem", "chain": chain.tolist(),
                                           "terminal_score": score}, "generations": generations}
                torch.cuda.synchronize()
                results[arm] = {"decision": output, "meter": meter.as_dict(),
                                "predictor_module_calls": core.calls-before_calls,
                                "wall_seconds": time.monotonic()-started, "evaluations": records}
            if before != {key: v4.model_receipt(model) for key, model in models.items()}:
                raise ValueError("model changed")
            output = {"scope": "model-only; exposed goal; RGB plus agent position/velocity",
                "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
                "v7_sha256": sha(Path(v7.__file__)), "v6_sha256": sha(Path(v6.__file__)),
                "adapter_sha256": sha(ROOT / "experiments/spectral_reachability/pc_jepa_history_rollout.py"),
                "manifest_sha256": sha(MANIFEST), "checkpoint_sha256": sha(ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"),
                "video_sha256": sha(video_path), "goal_rgb_sha256": v4.array_sha(goal_rgb),
                "history_rgb_sha256": saved["rgb_sha256"], "action_stats": stats,
                "chart_goal": chart_goal.tolist(), "chart_seed": 1, "gpu_uuid": uuid,
                "encoding_wall_seconds": encode_seconds, "strict_load": strict,
                "imported_sources": sources, "results": results, "physical_execution": False,
                "training": False, "pc3_pass": False, "pc5_pass": False}
    write_exclusive(OUTPUT, output)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha(OUTPUT),
        "scores": {k: v["decision"]["selected"]["terminal_score"] for k, v in results.items()},
        "meters": {k: v["meter"] for k, v in results.items()}}), flush=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        if not OUTPUT.exists() and not FAILURE.exists():
            write_exclusive(FAILURE, {"status": "failed", "exception": repr(exc),
                                      "traceback": traceback.format_exc()})
        raise
