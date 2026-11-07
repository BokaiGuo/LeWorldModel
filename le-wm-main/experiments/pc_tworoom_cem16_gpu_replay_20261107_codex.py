"""Re-score all sampled and incumbent candidates at replayed CEM16 decisions."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_cem16_variance_floor_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = source.base.ROOT.parent / "docs/PC_RGB_CEM16_GPU_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = source.base.ROOT / "results/pc_tworoom_cem16_gpu_replay_20261107_codex.json"


def rebuild_round(rng, mean, std, low, high):
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (5, 2) or std.shape != (5, 2) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("bad distribution")
    return np.clip(mean + rng.normal(size=(5, 5, 2)) * std, low, high)


def check_score_record(recomputed, recorded, selected, saved_chosen, *, tolerance=1e-3):
    scores = np.asarray(recomputed, dtype=np.float64)
    logged = np.asarray(recorded, dtype=np.float64)
    if (scores.shape != (16,) or logged.shape != (15,) or not np.isfinite(scores).all()
            or not np.isfinite(logged).all() or not np.isfinite(saved_chosen)
            or not 0 <= selected < 16 or tolerance < 0):
        raise ValueError("invalid scores")
    return {"max_sample_error": float(np.max(np.abs(scores[1:] - logged))),
            "selected_error": float(abs(scores[selected] - saved_chosen)),
            "recomputed_chosen": int(np.argmin(scores)),
            "selection_agrees": int(np.argmin(scores)) == selected,
            "within_tolerance": bool(np.max(np.abs(scores[1:] - logged)) <= tolerance
                                     and abs(scores[selected] - saved_chosen) <= tolerance)}


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(source.OUTPUT.read_text())
    if (data["runner_sha256"] != file_hash(source.__file__)
            or data["protocol_sha256"] != file_hash(source.PROTOCOL)
            or data["baseline_sha256"] != file_hash(source.base.OUTPUT)
            or data["input_sha256"] != file_hash(source.base.INPUT)):
        raise ValueError("source artifact drift")
    model, scaler, transform, normalizer = source.base.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != data["model_sha256"] or normalizer != data["normalizer_sha256"]:
        raise ValueError("checkpoint or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.OUTPUT, source.base.INPUT,
                                      source.base.common.CHECKPOINT / "weights.pt",
                                      source.base.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 TwoRoom dataset action scaler",
                         clock="five physical steps per model packet, fixed remaining deadline")
    low, high = (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(source.base.INPUT, allow_pickle=False) as inputs,
              h5py.File(source.base.common.DATA, "r") as dataset, torch.inference_mode()):
            for root, row in enumerate(data["rows"]):
                if (int(inputs["episodes"][root]), int(inputs["rows"][root])) != (row["episode"], row["row"]):
                    raise ValueError("task identity drift")
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                for arm in ("baseline", "floor"):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(dataset["proprio"][row["row"]])
                    actor._set_goal_state(dataset["proprio"][row["row"] + 25])
                    tail, cursor = None, 0
                    trace = row["arms"][arm]["trace"]
                    for iteration, decision in enumerate(row["arms"][arm]["decisions"]):
                        remaining = 25 - cursor
                        if remaining != decision["remaining"]:
                            raise ValueError("clock mismatch")
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(), "goal": goal,
                                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)
                        rng = np.random.default_rng(8140000 + row["episode"] * 10 + iteration)
                        initial = np.asarray(scaler.transform(source.base.cem.old_or_hold(tail)), dtype=np.float32)[None]
                        batches, recorded = [initial], []
                        for round_data in decision["rounds"]:
                            sample = rebuild_round(rng, round_data["mean"], round_data["std"], low, high)
                            batches.append(np.repeat(sample, 5, axis=1).astype(np.float32))
                            recorded.extend(round_data["costs"])
                        if tuple(map(len, batches)) != (1, 5, 5, 5):
                            raise ValueError("batch mismatch")
                        values = []
                        for batch in batches:
                            actions = torch.as_tensor(batch[:, :remaining].reshape(1, len(batch), remaining // 5, 10),
                                                      dtype=torch.float32, device="cuda")
                            endpoint, context = boundary.score(current, actions)
                            pred, target = context["predicted_emb"], context["goal_emb"][:, -1]
                            temporal = (pred[:, :, 1:] - target[:, None, None]).square().sum(-1)
                            if pred.shape[2] != remaining // 5 + 1 or not torch.allclose(
                                    endpoint, temporal[:, :, -1], atol=1e-3, rtol=0):
                                raise ValueError("bad model clock or endpoint")
                            values.extend(temporal.min(-1).values[0].cpu().tolist())
                        check = check_score_record(values, recorded, decision["chosen"], decision["chosen_cost"])
                        saved = np.asarray(decision["actions"], np.float32)
                        chosen_batch = np.concatenate(batches)[decision["chosen"]]
                        reconstructed = np.clip(scaler.inverse_transform(chosen_batch).astype(np.float32)[:remaining], -1, 1)
                        np.testing.assert_allclose(saved, reconstructed, atol=1e-6, rtol=0)
                        for action in decision["dispatched"]:
                            _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                            saved_trace = trace[cursor]
                            if (abs(info["distance_to_target"] - saved_trace["distance"]) > 1e-5
                                    or bool(term) != saved_trace["terminated"] or bool(trunc) != saved_trace["truncated"]):
                                raise ValueError("physical replay mismatch")
                            np.testing.assert_allclose(action, saved_trace["action"], rtol=0, atol=1e-6)
                            cursor += 1
                        tail = saved[len(decision["dispatched"]):]
                        rows.append({"episode": row["episode"], "arm": arm, "iteration": iteration,
                                     "scores": values, "recorded_choice": decision["chosen"], **check})
                print(f"episode={row['episode']} checked={len(rows)}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUT.open("x") as stream:
        json.dump({"scope": "same backend replay, not independent scientific confirmation",
                   "source_sha256": file_hash(source.OUTPUT), "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "rows": rows}, stream, allow_nan=False)
    print(f"sealed {len(rows)} decision audits", flush=True)


if __name__ == "__main__":
    main()
