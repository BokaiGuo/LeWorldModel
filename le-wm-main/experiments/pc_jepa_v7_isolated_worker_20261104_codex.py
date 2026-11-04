"""Model-only planner consuming an explicitly prepared, isolated input package."""
import hashlib
import json
import os
from pathlib import Path
import time

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
BASE = ROOT / "results/pc_jepa_v7_isolated_input_dev_20261104_codex"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
CAP, CHUNK = 1920, 4
UUID = "4e8973ff-0278-53f3-2bae-2615d461e441"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    output_path = BASE / "output/result.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    forbidden = [
        ROOT / "data/jepa_wm_osf/extracted/pusht_noise/val/obses/episode_001.mp4",
        ROOT / "data/jepa_wm_osf/extracted/pusht_noise/val/states.pth",
        ROOT / "data/jepa_wm_osf/extracted/pusht_noise/val/rel_actions.pth",
        ROOT / "results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json",
        ROOT / "results/pc_jepa_pusht_v7_planning_dev_20261104_codex.json",
    ]
    probes = {}
    for path in forbidden:
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except (FileNotFoundError, PermissionError) as exc:
            probes[str(path)] = type(exc).__name__
        else:
            raise RuntimeError(f"forbidden input readable: {path}")
    inp = BASE / "input"
    preparation = json.loads((inp / "preparation.json").read_text())
    if sha(inp / "context.npz") != preparation["input_sha256"] or sha(inp / "model_metadata.json") != preparation["model_metadata_sha256"]:
        raise ValueError("input digest mismatch")
    metadata = json.loads((inp / "model_metadata.json").read_text())
    if set(metadata) != {"official_weight_mapping", "expected_dynamics_schema", "expected_model_receipts", "seals"}:
        raise ValueError("unexpected metadata")
    expected = {"rgb_history": (3, 224, 224, 3), "goal_rgb": (1, 224, 224, 3),
                "normalized_proprio": (3, 4), "normalized_history_packets": (2, 10),
                "action_mean": (2,), "action_std": (2,)}
    with np.load(inp / "context.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("unexpected input fields")
        data = {key: archive[key].copy() for key in expected}
    if any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in expected.items()):
        raise ValueError("invalid input")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 required")
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    if uuid != UUID:
        raise RuntimeError("unexpected GPU")
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
            started = time.monotonic()
            def encode(images):
                pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                return models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
            context = encode(data["rgb_history"])
            goal = encode(data["goal_rgb"]).reshape(-1)
            visual = context.reshape(1, 3, 1, 16, 16, 384)
            prop = torch.from_numpy(data["normalized_proprio"]).unsqueeze(0).to("cuda")
            features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
            history = torch.from_numpy(data["normalized_history_packets"]).unsqueeze(0).to("cuda")
            mean = torch.as_tensor(data["action_mean"], device="cuda")
            std = torch.as_tensor(data["action_std"], device="cuda")
            if torch.any(std <= 0):
                raise ValueError("invalid action statistics")
            root = context[-1].reshape(-1)
            delta = goal - root
            norm = torch.linalg.vector_norm(delta)
            if norm.item() <= 1e-8:
                raise ValueError("degenerate chart")
            axis0 = delta / norm
            r = torch.as_tensor(np.random.default_rng(1).normal(size=axis0.numel()).astype(np.float32), device="cuda")
            r -= torch.dot(r, axis0) * axis0
            chart = torch.stack((axis0, r / torch.linalg.vector_norm(r)), dim=1) / np.sqrt(axis0.numel())
            chart_goal = (delta @ chart).cpu().numpy()
            torch.cuda.synchronize()
            encoding_seconds = time.monotonic() - started
            core = NativeCore(models["predictor"])
            results = {}
            for arm in ("v7_response", "smooth_cem"):
                meter, records = Meter(CAP), []
                before_calls = core.calls

                def evaluate(actions):
                    a = np.asarray(actions, dtype=np.float32)
                    if a.ndim != 3 or a.shape[1] not in (5, 10, 15) or a.shape[2] != 2 or not np.isfinite(a).all() or np.abs(a).max() > 1.000001:
                        raise ValueError("invalid trajectory")
                    a = np.clip(a, -1, 1)
                    def model_eval(sample):
                        positions, scores = [], []
                        for offset in range(0, len(sample), CHUNK):
                            chunk = torch.as_tensor(sample[offset:offset + CHUNK], device="cuda")
                            packet = ((chunk - mean) / std).reshape(len(chunk), -1, 10).transpose(0, 1)
                            pred = unroll_history(core, visual, features, history, packet, 3)["visual"][-1]
                            flat = pred.reshape(len(chunk), -1)
                            positions.append(((flat - root) @ chart).cpu().numpy())
                            scores.append((flat - goal).square().mean(dim=1).cpu().numpy())
                        torch.cuda.synchronize()
                        return np.concatenate(positions), np.concatenate(scores)
                    positions, scores = meter.predict(a, model_eval)
                    records.append({"actions": a.tolist(), "positions": positions.tolist(), "scores": scores.tolist()})
                    if len(records) % 30 == 0:
                        print(f"{arm}: calls={len(records)} tokens={meter.token_samples}/{CAP}", flush=True)
                    return positions, scores

                started = time.monotonic()
                if arm == "v7_response":
                    decision = v7.decide("response", "response", 0, 0, chart_goal,
                                         evaluate, meter, temporal.NOMINALS, None)
                else:
                    initial = np.zeros((15, 2))
                    initial_score = float(evaluate(initial[None])[1][0])
                    count = (CAP - meter.token_samples) // 3
                    chain, score, generations = v6.cem(initial, [], count,
                        np.random.default_rng(2027110401), evaluate, initial_score, 4)
                    decision = {"selected": {"name": "smooth_cem", "chain": chain.tolist(),
                                              "terminal_score": score}, "generations": generations}
                torch.cuda.synchronize()
                results[arm] = {"decision": decision, "meter": meter.as_dict(),
                    "predictor_module_calls": core.calls - before_calls,
                    "wall_seconds": time.monotonic() - started, "evaluations": records}
            if before != {key: v4.model_receipt(model) for key, model in models.items()}:
                raise ValueError("model changed")
    output = {"scope": "isolated model-only development regression; exposed goal; RGB plus proprioception",
        "preparation": preparation, "worker_sha256": sha(Path(__file__)), "forbidden_path_probes": probes,
        "checkpoint_sha256": sha(ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar"),
        "v7_sha256": sha(Path(v7.__file__)), "v6_sha256": sha(Path(v6.__file__)),
        "chart_goal": chart_goal.tolist(), "chart_seed": 1, "gpu_uuid": uuid,
        "encoding_wall_seconds": encoding_seconds, "strict_load": strict, "imported_sources": sources,
        "results": results, "physical_execution": False, "training": False,
        "pc3_pass": False, "pc5_pass": False}
    with output_path.open("x") as stream:
        json.dump(output, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(f"wrote {output_path} sha256={sha(output_path)}", flush=True)


if __name__ == "__main__":
    main()
