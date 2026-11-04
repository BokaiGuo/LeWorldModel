"""Post-hoc frozen-model diagnostic for the six actual native PushT packets."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_native_v7_feedback_dev_20261104_codex"
OUT = ROOT / "results/pc_jepa_native_executed_packet_diag_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_NATIVE_EXECUTED_PACKET_DIAG_PROTOCOL_20261104_CODEX.md"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
UUID = "4e8973ff-0278-53f3-2bae-2615d461e441"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def metrics(current, predicted, actual, goal):
    current, predicted, actual, goal = (np.asarray(x, dtype=np.float64).reshape(-1)
                                        for x in (current, predicted, actual, goal))
    mse = lambda x, y: float(np.mean((x - y) ** 2))
    d_pred, d_actual = predicted - current, actual - current
    p, a = np.linalg.norm(d_pred), np.linalg.norm(d_actual)
    return {"pred_vs_actual_mse": mse(predicted, actual),
            "persistence_vs_actual_mse": mse(current, actual),
            "current_goal_mse": mse(current, goal), "predicted_goal_mse": mse(predicted, goal),
            "actual_goal_mse": mse(actual, goal),
            "predicted_goal_change": mse(predicted, goal) - mse(current, goal),
            "actual_goal_change": mse(actual, goal) - mse(current, goal),
            "predicted_displacement_l2": float(p), "actual_displacement_l2": float(a),
            "displacement_error_l2": float(np.linalg.norm(d_pred - d_actual)),
            "displacement_cosine": float(np.dot(d_pred, d_actual) / (p * a)) if min(p, a) > 1e-12 else None}


def main():
    print("Compare the six dispatched packets against their actual native successor RGB; post-hoc only.", flush=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads((SOURCE / "audit.json").read_text())
    physical = json.loads((SOURCE / "physical_replay.json").read_text())
    summary = SOURCE / "summary.json"
    if audit["status"] != "saved_data_audit_passed" or audit["summary_sha256"] != sha(summary):
        raise ValueError("saved audit mismatch")
    if physical["status"] != "exact" or physical["summary_sha256"] != sha(summary):
        raise ValueError("physical replay mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("GPU 0 required")
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    if uuid != UUID:
        raise ValueError("unexpected GPU")
    bindings = {str(p): sha(p) for p in [PROTOCOL, Path(__file__), summary,
                SOURCE / "manifest.json", SOURCE / "audit.json", SOURCE / "physical_replay.json"]}
    OUT.mkdir()
    result = {"scope": "post-hoc six executed packets, exposed root; no physical execution or planning",
              "bindings": bindings, "gpu_uuid": uuid, "rows": [], "status": "running",
              "training": False, "pc_gate_pass": False}
    error = None
    try:
        with gpu_lock(LOCK):
            meta_path = SOURCE / "response_0/input/model_metadata.json"
            metadata = json.loads(meta_path.read_text())
            models, strict, sources, before = native.load_cuda(ROOT, metadata)
            _, v5, _ = native.helper(ROOT).dependencies(ROOT)
            v4 = v5.import_v4(ROOT)
            from app.plan_common.datasets.transforms import make_transforms
            from torch.nn.attention import SDPBackend, sdpa_kernel
            transform = make_transforms(img_size=224, random_horizontal_flip=False,
                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                reprob=0., auto_augment=False, motion_shift=False)
            core = NativeCore(models["predictor"])
            started = time.monotonic()
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for arm in ("response", "cem"):
                    arm_report = json.loads((SOURCE / f"{arm}_result.json").read_text())
                    for step in range(3):
                        directory = SOURCE / f"{arm}_{step}"
                        input_path, decision_path = directory / "input/context.npz", directory / "output/decision.json"
                        actual_path = SOURCE / f"{arm}_physical/frame_{15 + 5 * step:02d}/rgb.npy"
                        for path in (input_path, decision_path, actual_path):
                            bindings[str(path)] = sha(path)
                        decision = json.loads(decision_path.read_text())
                        if decision["input_sha256"] != sha(input_path):
                            raise ValueError("decision input mismatch")
                        actions = np.asarray(decision["decision"]["selected"]["chain"], np.float32)
                        if actions.shape != (15 - 5 * step, 2):
                            raise ValueError("action horizon mismatch")
                        for i, action in enumerate(actions[:5], start=11 + 5 * step):
                            command = json.loads((SOURCE / f"{arm}_physical/command_{i:02d}.json").read_text())
                            if command["decision_sha256"] != sha(decision_path) or not np.array_equal(
                                    action, np.asarray(command["action"], np.float32)):
                                raise ValueError("actual dispatched action mismatch")
                        with np.load(input_path, allow_pickle=False) as f:
                            context = {k: f[k].copy() for k in f.files}
                        actual_rgb = np.load(actual_path, allow_pickle=False)
                        images = np.concatenate((context["rgb_history"], context["goal_rgb"], actual_rgb[None]))
                        pixels = torch.from_numpy(images).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                        patch = models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
                        visual = patch[:3].reshape(1, 3, 1, 16, 16, 384)
                        prop = torch.from_numpy(context["normalized_proprio"]).unsqueeze(0).to("cuda")
                        features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
                        history = torch.from_numpy(context["normalized_history_packets"]).unsqueeze(0).to("cuda")
                        mean = torch.as_tensor(context["action_mean"], device="cuda")
                        std = torch.as_tensor(context["action_std"], device="cuda")
                        normalized = (torch.from_numpy(actions).to("cuda") - mean) / std
                        packet = normalized.reshape(-1, 1, 10)
                        full = unroll_history(core, visual, features, history, packet, 3)["visual"]
                        predicted = full[3, 0].reshape(256, 384)
                        final = full[-1, 0].reshape(256, 384)
                        arr = {"current": patch[2].cpu().numpy(), "predicted": predicted.cpu().numpy(),
                               "actual": patch[4].cpu().numpy(), "goal": patch[3].cpu().numpy(),
                               "full_terminal": final.cpu().numpy()}
                        path = OUT / f"{arm}_{step}.npz"
                        with path.open("xb") as f:
                            np.savez_compressed(f, **arr)
                        original = float(decision["decision"]["selected"]["terminal_score"])
                        rerun = float((final - patch[3]).square().mean().item())
                        row = {"arm": arm, "step": step, "frame": 10 + 5 * step,
                               "actual_frame": 15 + 5 * step, "selected_score_original": original,
                               "full_terminal_score_recomputed_float32": rerun,
                               "full_terminal_score_abs_error": abs(original - rerun),
                               "metrics": metrics(arr["current"], arr["predicted"], arr["actual"], arr["goal"]),
                               "physical_before": arm_report["events"][10 + 5 * step]["metrics"],
                               "physical_after": arm_report["events"][15 + 5 * step]["metrics"],
                               "array_file": path.name, "array_sha256": sha(path),
                               "decision_sha256": sha(decision_path), "input_sha256": sha(input_path),
                               "actual_rgb_sha256": sha(actual_path)}
                        result["rows"].append(row)
                        print(f"{arm} step={step} prediction_mse={row['metrics']['pred_vs_actual_mse']:.6f} "
                              f"persistence_mse={row['metrics']['persistence_vs_actual_mse']:.6f} "
                              f"score_error={row['full_terminal_score_abs_error']:.3g}", flush=True)
            torch.cuda.synchronize()
            result.update(wall_seconds=time.monotonic()-started, predictor_module_calls=core.calls,
                          strict_load=strict, imported_sources=sources,
                          model_unchanged=before == {k: v4.model_receipt(m) for k, m in models.items()})
            if not result["model_unchanged"]:
                raise ValueError("model changed")
            if any(sha(Path(p)) != v for p, v in bindings.items()):
                raise ValueError("source changed")
            result["status"] = "completed"
    except Exception:
        error = traceback.format_exc()
        result["status"] = "failed"
    result["error"] = error
    with (OUT / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
