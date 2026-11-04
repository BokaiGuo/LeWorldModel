"""Frozen RGB model replay against sealed decision traces and unified compiler."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_multidirection_rgb_fixed_terminal_dev_20261103_codex as policy
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history
from experiments.spectral_reachability import pc_unified_multidirection_dev_v1 as unified
from experiments import pc_budget_allocation_20261103_codex as allocation
from experiments.spectral_reachability import pc_spectral_bank_planner_20261103_codex as meter_module

run, old, h, ROOT = policy.runner, policy.runner.old, policy.runner.h, policy.runner.ROOT
PROTOCOL = ROOT.parent / "docs/PC_UNIFIED_RGB_FROZEN_MODEL_REPLAY_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_unified_rgb_frozen_model_replay_20261103_codex.json"
TOL_PRED, TOL_ACTION = 2e-5, 1e-5


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    run.PREFIX = policy.PREFIX
    sha = old.prior.sha
    files = [Path(__file__), PROTOCOL, Path(unified.__file__), Path(policy.__file__),
             Path(run.__file__), Path(unroll_with_action_history.__code__.co_filename),
             Path(allocation.__file__), Path(meter_module.__file__), h.DEFAULT_CHECKPOINT,
             h.DEFAULT_CONFIG, old.HISTORY, old.DECISIONS, h.CAMERA, h.OLD, h.OBSTACLE.with_suffix(".npz")]
    for root in range(4):
        decisions, execution, images = run.paths(root, "multidirection")
        files.extend([*decisions, execution, images])
    sources = {str(p): sha(p) for p in files}
    baseline = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != baseline["history_sha256"]:
        raise ValueError("history archive mismatch")
    camera, calibration = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("Frozen model: replay sealed RGB actions and recompile twelve saved contexts.", flush=True)
    started = time.monotonic()
    rows = []
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != baseline["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != baseline["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), baseline["action_mean"]) or not np.array_equal(std.cpu().numpy(), baseline["action_std"]):
            raise ValueError("normalizer mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != baseline["goal_sha256"]:
            raise ValueError("goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            goal_position = ((target.mean(0) @ pinv.T)[:2] * pos_std).cpu().numpy()
        for root in range(4):
            decisions, execution, images = run.paths(root, "multidirection")
            receipt = json.loads(execution.read_text())
            if receipt["images_sha256"] != sha(images):
                raise ValueError("execution images mismatch")
            with np.load(images, allow_pickle=False) as archive:
                frames = archive["rgb"]
                if len(frames) != 31:
                    raise ValueError("invalid frame clock")
                for step in range(3):
                    decision = json.loads(decisions[step].read_text())
                    saved = decision["result"]
                    ticks = decision["ticks"]
                    if (receipt["rows"][step]["decision_sha256"] != sha(decisions[step])
                            or decision["input_rgb_sha256"] != [old.digest_image(frames[t]) for t in ticks]
                            or decision["model_hash"] != model_hash
                            or saved["actual_source_sha256"] != policy.source_hashes()):
                        raise ValueError("source or context mismatch")
                    with torch.no_grad():
                        entries = [h.encode_with_state(model, frames[t],
                                   h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                                   for t in ticks]
                    ctx = {key: torch.cat([entry[key] for entry in entries], dim=1) for key in ("visual", "proprio")}
                    history = ((torch.as_tensor(decision["history_actions"], dtype=torch.float32, device=device) - mean) / std).reshape(2, 1, 10)

                    def predict(chains):
                        batch = np.asarray(chains, dtype=np.float32)
                        n, steps, _ = batch.shape
                        packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device) - mean) / std).reshape(n, steps // 5, 10).transpose(0, 1)
                        with torch.no_grad():
                            out = unroll_with_action_history(model, ctx, history.expand(2, n, 10), packets, 3)["proprio"][-1]
                            features = out.reshape(n, 256, 16)
                            pos = (features.mean(1) @ pinv.T)[:, :2] * pos_std
                            score = h.projected_cost(features, target, pinv, pos_std)
                            torch.cuda.synchronize()
                        return pos.cpu().numpy(), score.cpu().numpy()

                    original_trace = saved["prediction_trace"]
                    fixed = []
                    for index, batch in enumerate(original_trace):
                        actions = np.asarray(batch["model_raw_float32_actions"], dtype=np.float32)
                        positions, scores = predict(actions)
                        pos_err = float(np.max(np.abs(positions.astype(float) - np.asarray(batch["positions"]))))
                        score_err = float(np.max(np.abs(scores.astype(float) - np.asarray(batch["scores"]))))
                        fixed.append({"index": index, "count": len(actions), "horizon": actions.shape[1],
                                      "position_max_abs": pos_err, "score_max_abs": score_err,
                                      "within_tolerance": bool(pos_err <= TOL_PRED and score_err <= TOL_PRED)})
                    carry = None if step == 0 else np.asarray(saved["carry"], dtype=float)
                    context_id = sha(decisions[step])
                    compiled = unified.compile_decision(root=root, step=step, target=goal_position,
                                                        predict=predict, context_id=context_id, carry=carry)
                    chain = np.asarray(compiled["core_result"]["selected"]["chain"])
                    original = np.asarray(saved["selected"]["chain"])
                    action_error = float(np.max(np.abs(chain - original)))
                    score_error = abs(float(compiled["core_result"]["selected"]["terminal_score"]) - float(saved["selected"]["terminal_score"]))
                    paired = []
                    for index, (new, previous) in enumerate(zip(compiled["prediction_trace"], original_trace)):
                        a, b = np.asarray(new["model_raw_float32_actions"]), np.asarray(previous["model_raw_float32_actions"])
                        if a.shape != b.shape:
                            paired.append({"index": index, "shape_match": False})
                            continue
                        paired.append({"index": index, "shape_match": True,
                                       "action_max_abs": float(np.max(np.abs(a-b))),
                                       "position_max_abs": float(np.max(np.abs(np.asarray(new["positions"]) - previous["positions"]))),
                                       "score_max_abs": float(np.max(np.abs(np.asarray(new["scores"]) - previous["scores"])))})
                    rows.append({"root": root, "step": step, "decision_sha256": context_id,
                                 "ticks": ticks, "input_rgb_sha256": decision["input_rgb_sha256"],
                                 "history_actions": decision["history_actions"], "fixed_batches": fixed,
                                 "fixed_tokens": sum(x["count"] * x["horizon"] // 5 for x in fixed),
                                 "recompiled": compiled, "batch_comparison": paired,
                                 "same_batch_count": len(compiled["prediction_trace"]) == len(original_trace),
                                 "selected_action_max_abs": action_error, "selected_score_abs": score_error,
                                 "selected_within_tolerance": bool(action_error <= TOL_ACTION and score_error <= TOL_PRED),
                                 "abstention_matches": compiled["core_result"]["abstained"] == saved["abstained"]})
                    print(f"root={root} step={step} fixed={sum(x['within_tolerance'] for x in fixed)}/{len(fixed)} "
                          f"selected_error={action_error:.3g}", flush=True)
        torch.cuda.synchronize()
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model changed")
    if len(rows) != 12 or sources != {str(p): sha(p) for p in files}:
        raise ValueError("panel or dependency drift")
    payload = {"scope": "frozen observed RGB replay, no new physical execution", "dependencies": sources,
               "model_hash": model_hash, "goal_position": goal_position.tolist(),
               "action_mean": mean.cpu().tolist(), "action_std": std.cpu().tolist(),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
               "tolerances": {"prediction_abs": TOL_PRED, "selected_action_abs": TOL_ACTION},
               "encoding_calls": 1 + 12 * 3, "elapsed_seconds": time.monotonic() - started, "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    print("Frozen RGB replay sealed.", flush=True)


if __name__ == "__main__":
    main()
