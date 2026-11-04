"""Attribute saved-score roundoff to encoder and predictor batch shapes."""
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.pc_jepa_history_rollout_dev_20261104_codex import NativeCore
from experiments.pc_jepa_native_executed_packet_diag_20261104_codex import ROOT, SOURCE, LOCK, UUID, sha
from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as native
from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
from experiments.spectral_reachability.pc_jepa_history_rollout import unroll_history

OUT = ROOT / "results/pc_jepa_score_replay_precision_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_SCORE_REPLAY_PRECISION_PROTOCOL_20261104_CODEX.md"
DIAG = ROOT / "results/pc_jepa_native_executed_packet_diag_20261104_codex"


def main():
    print("Cross encoder and predictor batch shapes for all six previously selected chains.", flush=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    diag = json.loads((DIAG / "result.json").read_text())
    audit = json.loads((DIAG / "audit.json").read_text())
    if diag["status"] != "completed" or audit["result_sha256"] != sha(DIAG / "result.json"):
        raise ValueError("prior diagnostic audit mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available() or (
            str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") != UUID):
        raise ValueError("unexpected GPU")
    OUT.mkdir()
    report = {"scope": "post-hoc numerical precision diagnostic; no physical or planning actions",
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "diagnostic_sha256": sha(DIAG / "result.json"), "audit_sha256": sha(DIAG / "audit.json"),
              "rows": [], "status": "running", "pc_gate_pass": False}
    error = None
    try:
        with gpu_lock(LOCK):
            meta = json.loads((SOURCE / "response_0/input/model_metadata.json").read_text())
            models, strict, sources, before = native.load_cuda(ROOT, meta)
            _, v5, _ = native.helper(ROOT).dependencies(ROOT)
            v4 = v5.import_v4(ROOT)
            from app.plan_common.datasets.transforms import make_transforms
            from torch.nn.attention import SDPBackend, sdpa_kernel
            transform = make_transforms(img_size=224, random_horizontal_flip=False,
                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                reprob=0., auto_augment=False, motion_shift=False)
            core = NativeCore(models["predictor"])
            start = time.monotonic()
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                def encode(images):
                    pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255.
                    return models["official_encoder"](transform(pixels), is_training=True)["x_norm_patchtokens"]
                for arm in ("response", "cem"):
                    for step in range(3):
                        folder = SOURCE / f"{arm}_{step}"
                        decision = json.loads((folder / "output/decision.json").read_text())
                        chain = np.asarray(decision["decision"]["selected"]["chain"], np.float32)
                        score = decision["decision"]["selected"]["terminal_score"]
                        chosen = None
                        for b, entry in enumerate(decision["evaluations"]):
                            actions = np.asarray(entry["actions"], np.float32)
                            if actions.shape[1:] != chain.shape:
                                continue
                            for i, (action, candidate_score) in enumerate(zip(actions, entry["scores"])):
                                if np.array_equal(action, chain) and candidate_score == score:
                                    start_index = (i // 4) * 4
                                    chosen = (b, i, actions[start_index:start_index+4], i-start_index)
                                    break
                            if chosen is not None:
                                break
                        if chosen is None:
                            raise ValueError("selected action/score not found in original batch")
                        batch_index, candidate_index, chunk, local_index = chosen
                        with np.load(folder / "input/context.npz", allow_pickle=False) as f:
                            data = {k: f[k].copy() for k in f.files}
                        actual = np.load(SOURCE / f"{arm}_physical/frame_{15+5*step:02d}/rgb.npy", allow_pickle=False)
                        split_context = encode(data["rgb_history"])
                        split_goal = encode(data["goal_rgb"])
                        joined = encode(np.concatenate((data["rgb_history"], data["goal_rgb"], actual[None])))
                        feature_max = float((torch.cat((split_context, split_goal), dim=0)-joined[:4]).abs().max().item())
                        history = torch.from_numpy(data["normalized_history_packets"]).unsqueeze(0).to("cuda")
                        prop = torch.from_numpy(data["normalized_proprio"]).unsqueeze(0).to("cuda")
                        features = models["proprio_encoder"](prop).expand(1, 3, 256, 16)
                        mean, std = (torch.as_tensor(data[k], device="cuda") for k in ("action_mean", "action_std"))
                        values = {}
                        for encoding, (ctx, goal) in {
                                "split": (split_context, split_goal), "joined5": (joined[:3], joined[3:4])}.items():
                            visual = ctx.reshape(1, 3, 1, 16, 16, 384)
                            for batch, (action_batch, pick) in {"original_chunk": (chunk, local_index),
                                                                 "single": (chain[None], 0)}.items():
                                t = torch.as_tensor(action_batch, device="cuda")
                                packet = ((t - mean) / std).reshape(len(t), -1, 10).transpose(0, 1)
                                pred = unroll_history(core, visual, features, history, packet, 3)["visual"][-1]
                                value = (pred.reshape(len(t), -1) - goal.reshape(-1)).square().mean(dim=1)[pick].item()
                                values[f"{encoding}_{batch}"] = {"score": value, "abs_difference_from_record": abs(value-score)}
                        row = {"arm": arm, "step": step, "recorded_score": score,
                               "original_batch_index": batch_index, "candidate_index": candidate_index,
                               "original_chunk_size": len(chunk), "selected_chunk_index": local_index,
                               "encoder_max_absolute_difference": feature_max, "combinations": values,
                               "decision_sha256": sha(folder / "output/decision.json"),
                               "input_sha256": sha(folder / "input/context.npz")}
                        report["rows"].append(row)
                        print(f"{arm} {step} split/original gap={values['split_original_chunk']['abs_difference_from_record']:.3g} "
                              f"joined/single gap={values['joined5_single']['abs_difference_from_record']:.3g}", flush=True)
            torch.cuda.synchronize()
            report.update(status="completed", wall_seconds=time.monotonic()-start, strict_load=strict,
                          imported_sources=sources, predictor_module_calls=core.calls,
                          model_unchanged=before == {k: v4.model_receipt(m) for k, m in models.items()})
            if not report["model_unchanged"]:
                raise ValueError("model changed")
    except Exception:
        error = traceback.format_exc()
        report["status"] = "failed"
    report["error"] = error
    with (OUT / "result.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
