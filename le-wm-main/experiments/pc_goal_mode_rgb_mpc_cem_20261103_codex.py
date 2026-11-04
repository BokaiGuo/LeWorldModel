"""Direct-CEM real-RGB counterpart to the goal-mode MPC smoke."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_goal_mode_abstain_rgb_mpc_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import ShadowRGBBridge
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, base, ROOT = prior.old, prior.h, prior.base, prior.ROOT
PREFIX = ROOT / "results/pc_goal_mode_rgb_mpc_cem_20261103_codex"
PROTOCOL = ROOT.parent / "docs/PC_GOAL_MODE_RGB_MPC_CEM_DEV_PROTOCOL_20261103_CODEX.md"
DECISIONS = [PREFIX.with_name(PREFIX.name + f"_decision{i}.json") for i in range(3)]
EXECUTION = PREFIX.with_name(PREFIX.name + "_execution.json")
IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")


def main():
    if any(path.exists() for path in (*DECISIONS, EXECUTION, IMAGES)):
        raise FileExistsError("refusing to overwrite full or partial execution")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history archive mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("Three real-RGB direct-CEM decisions; matched-token exploratory baseline.", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        started = time.monotonic()
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if (not np.array_equal(mean.cpu().numpy(), source["action_mean"])
                or not np.array_equal(std.cpu().numpy(), source["action_std"])):
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("RGB goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
            torch.cuda.synchronize()
        setup_seconds = time.monotonic() - started
        rows = []
        with ShadowRGBBridge(np.asarray([*old.STARTS[0], 0., 0.])) as env:
            warm, _ = old.do_actions(env, old.WARMUP)
            frames = [env.initial_rgb.copy(), *warm]
            if ([old.digest_image(x) for x in frames] != source["rows"][0]["rgb_sha256"]
                    or not np.array_equal(np.stack(frames), histories["root0_history_rgb"])):
                raise ValueError("warmup mismatch")
            actions = old.WARMUP.tolist()
            for step in range(3):
                ticks = prior.context_ticks(len(frames))
                started = time.monotonic()
                with torch.no_grad():
                    entries = [h.encode_with_state(model, frames[t],
                               h.causal_state(frames[t - 1], frames[t], metadata, offset, .05), device)
                               for t in ticks]
                    torch.cuda.synchronize()
                ctx = {key: torch.cat([entry[key] for entry in entries], dim=1)
                       for key in ("visual", "proprio")}
                history_actions = np.asarray(actions[-10:], dtype=float)
                history = ((torch.as_tensor(history_actions, dtype=torch.float32, device=device)
                            - mean) / std).reshape(2, 1, 10)
                encoding_seconds = time.monotonic() - started
                meter = base.Meter(132)

                def scores(chains):
                    def raw(batch):
                        n, steps, _ = batch.shape
                        packets = ((torch.as_tensor(batch, dtype=torch.float32, device=device) - mean) / std).reshape(
                            n, steps // 5, 10).transpose(0, 1)
                        with torch.no_grad():
                            pred = unroll_with_action_history(model, ctx, history.expand(2, n, 10),
                                                              packets, 3)["proprio"][-1]
                            value = h.projected_cost(pred.reshape(n, 256, 16), target, pinv, pos_std)
                            torch.cuda.synchronize()
                        return value.cpu().numpy()
                    return meter.predict(np.asarray(chains, dtype=np.float32), raw)

                started = time.monotonic()
                chain, value, scores_saved = optimize([], 15, 44, np.random.default_rng(2026110710 + step), scores)
                if meter.token_samples != 132:
                    raise ValueError("prediction budget mismatch")
                decision = {"step": step, "ticks": ticks, "history_actions": history_actions.tolist(),
                            "input_rgb_sha256": [old.digest_image(frames[t]) for t in ticks],
                            "chain": chain.tolist(), "terminal_score": value, "scores": scores_saved,
                            "token_counts": meter.as_dict(), "encoding_calls": 3,
                            "encoding_seconds": encoding_seconds, "planning_seconds": time.monotonic() - started,
                            "model_hash": model_hash, "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT),
                            "history_sha256": sha(old.HISTORY), "protocol_sha256": sha(PROTOCOL),
                            "runner_sha256": sha(Path(__file__)), "source_sha256": sha(old.DECISIONS),
                            "camera_sha256": sha(h.CAMERA), "calibration_sha256": sha(h.OLD),
                            "goal_sha256": old.digest_image(goal), "setup_seconds_shared": setup_seconds,
                            "full_cost_certified": False}
                old.write_json(DECISIONS[step], decision)
                observed, times = old.do_actions(env, chain[:5])
                frames.extend(observed)
                actions.extend(chain[:5].tolist())
                rows.append({"step": step, "decision_sha256": sha(DECISIONS[step]),
                             "actions": chain[:5].tolist(), "rgb_sha256": [
                                 old.digest_image(image) for image in observed], "times_ns": times})
                print(f"decision {step}: 44 full-horizon CEM samples", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model changed")
        with IMAGES.open("xb") as file:
            np.savez_compressed(file, rgb=np.stack(frames))
        old.write_json(EXECUTION, {"rows": rows, "images_sha256": sha(IMAGES),
                                   "executed_actions": actions[15:], "full_cost_certified": False})
        print("executed 15 actions across three real-RGB CEM decisions", flush=True)


if __name__ == "__main__":
    main()
