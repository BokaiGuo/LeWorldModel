"""Replay every saved RGB model prediction without changing the original run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_successor_rgb_execution_dev_20261103_codex as pilot
from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

ROOT, h, old = runner.ROOT, runner.h, runner.old
PROTOCOL = ROOT.parent / "docs/PC_SUCCESSOR_RGB_MODEL_TRACE_REPLAY_AUDIT_PROTOCOL_20261103_CODEX.md"
OUTPUT = pilot.PREFIX.with_name(pilot.PREFIX.name + "_model_trace_replay.json")
TOLERANCE = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(old.DECISIONS.read_text())
    camera, calibration = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    dependencies = [Path(__file__), PROTOCOL, Path(pilot.__file__), Path(runner.__file__),
                    Path(pilot.policy.__file__), Path(pilot.policy.generator.__file__),
                    Path(unroll_with_action_history.__code__.co_filename), h.DEFAULT_CHECKPOINT,
                    h.CAMERA, h.OLD, old.DECISIONS, old.HISTORY]
    hashes = {str(p): sha(p) for p in dependencies}
    print("Replaying every saved RGB model batch; development audit, no new selection.", flush=True)
    runner.PROTOCOL, runner.PREFIX, runner.ARMS = pilot.PROTOCOL, pilot.PREFIX, pilot.ARMS
    with np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (not np.array_equal(mean.cpu().numpy(), source["action_mean"])
                or not np.array_equal(std.cpu().numpy(), source["action_std"])):
            raise ValueError("normalizer mismatch")
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
        rows = []
        for root in range(4):
            for arm in pilot.ARMS:
                decisions, execution, images = runner.paths(root, arm)
                receipt = json.loads(execution.read_text())
                if receipt["images_sha256"] != sha(images):
                    raise ValueError("image receipt mismatch")
                with np.load(images, allow_pickle=False) as data:
                    frames = data["rgb"]
                    for step, path in enumerate(decisions):
                        decision = json.loads(path.read_text())
                        result = decision["result"]
                        if (receipt["rows"][step]["decision_sha256"] != sha(path)
                                or result["policy_sha256"] != sha(Path(pilot.__file__))
                                or decision["model_hash"] != model_hash
                                or decision["input_rgb_sha256"] != [old.digest_image(frames[t]) for t in decision["ticks"]]):
                            raise ValueError("decision receipt mismatch")
                        with torch.no_grad():
                            entries = [h.encode_with_state(model, frames[t],
                                       h.causal_state(frames[t-1], frames[t], metadata, offset, .05), device)
                                       for t in decision["ticks"]]
                        ctx = {key: torch.cat([entry[key] for entry in entries], dim=1)
                               for key in ("visual", "proprio")}
                        history = ((torch.as_tensor(decision["history_actions"], dtype=torch.float32, device=device)
                                    - mean) / std).reshape(2, 1, 10)
                        pos_max = score_max = 0.
                        tokens = samples = failures = 0
                        for batch in result["prediction_trace"]:
                            actions = np.asarray(batch["model_actions"], dtype=np.float32)
                            if (actions.ndim != 3 or actions.shape[2] != 2 or actions.shape[1] not in (5, 10, 15)
                                    or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1 + 1e-6)):
                                raise ValueError("invalid recorded action")
                            n, steps, _ = actions.shape
                            packets = ((torch.as_tensor(actions, device=device) - mean) / std).reshape(
                                n, steps // 5, 10).transpose(0, 1)
                            with torch.no_grad():
                                pred = unroll_with_action_history(model, ctx, history.expand(2, n, 10),
                                                                  packets, 3)["proprio"][-1]
                                features = pred.reshape(n, 256, 16)
                                positions = ((features.mean(1) @ pinv.T)[:, :2] * pos_std).cpu().numpy()
                                scores = h.projected_cost(features, target, pinv, pos_std).cpu().numpy()
                            if not np.isfinite(positions).all() or not np.isfinite(scores).all():
                                raise ValueError("nonfinite replay output")
                            pd = float(np.max(np.abs(positions - np.asarray(batch["positions"]))))
                            sd = float(np.max(np.abs(scores - np.asarray(batch["scores"]))))
                            pos_max, score_max = max(pos_max, pd), max(score_max, sd)
                            failures += int(pd > TOLERANCE or sd > TOLERANCE)
                            tokens += n * steps // 5
                            samples += n
                        if (tokens != decision["counts"]["prediction_token_samples"]
                                or samples != decision["counts"]["trajectory_samples"]):
                            raise ValueError("prediction count mismatch")
                        rows.append({"root": root, "arm": arm, "step": step, "decision_sha256": sha(path),
                                     "batches": len(result["prediction_trace"]), "samples": samples,
                                     "tokens": tokens, "max_position_abs_error": pos_max,
                                     "max_score_abs_error": score_max, "failed_batches": failures})
                print(f"root {root} {arm}: three decisions replayed", flush=True)
        if h.parameter_hash(model) != model_hash or hashes != {str(p): sha(p) for p in dependencies}:
            raise ValueError("audit-time dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "tolerance": TOLERANCE, "audit_time_dependency_sha256": hashes,
                   "scope": "post-hoc engineering replay; not original-run dependency certification",
                   "failed_batches": sum(x["failed_batches"] for x in rows)},
                  stream, indent=2, allow_nan=False)
    print(f"Replayed {len(rows)} decisions; failed batches: {sum(x['failed_batches'] for x in rows)}.", flush=True)


if __name__ == "__main__":
    main()
