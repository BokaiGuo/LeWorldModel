"""Replay decisions from RGB observations and action traces with the same model."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_incumbent_rgb_feedback_20261107_codex as run
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUT = run.ROOT / "results/pc_tworoom_incumbent_rgb_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    record = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("planner_sha256", run.planner.__file__)):
        if record[key] != file_hash(Path(path)):
            raise ValueError(f"binding drift {key}")
    common = run.prior.prior.source.block.prior.common
    input_path = run.prior.prior.source.block.prior.inputs.INPUT
    if record["input_sha256"] != file_hash(input_path):
        raise ValueError("input drift")
    model, scaler, transform, normalizer = common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != record["model_sha256"] or normalizer != record["normalizer_sha256"]:
        raise ValueError("model/normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), run.PROTOCOL, Path(run.planner.__file__),
                         input_path, common.CHECKPOINT / "weights.pt"],
                         preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="remaining deadline, five steps per packet")
    max_error = 0.0
    checked = 0
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    try:
        with (np.load(input_path, allow_pickle=False) as inputs,
              h5py.File(common.DATA, "r") as data, torch.inference_mode()):
            for task in record["rows"]:
                root, row = task["root"], task["row"]
                goal = transform(inputs["goal"][root])[None, None, None].cuda()
                for arm, episode in task["arms"].items():
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row + 25])
                    cursor, old_tail = 0, None
                    for decision in episode["decisions"]:
                        remaining = 25 - cursor
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)

                        def score(actions):
                            normal = scaler.transform(actions.reshape(-1, 2)).reshape(-1, remaining // 5, 10)
                            values = []
                            for offset in range(0, len(actions), 32):
                                chunk = torch.as_tensor(normal[offset:offset + 32][None],
                                                        device="cuda", dtype=torch.float32)
                                value, _ = boundary.score(current, chunk)
                                values.extend(value[0].cpu().tolist())
                            return np.asarray(values)

                        replay = run.planner.search(score, remaining=remaining, old_tail=old_tail,
                                                    seed=decision["seed"], arm=arm)
                        expected = np.asarray(decision["all_scores"])
                        max_error = max(max_error, float(np.max(np.abs(replay["all_scores"] - expected))))
                        if not np.allclose(replay["all_scores"], expected, rtol=1e-5, atol=1e-4):
                            raise ValueError("score mismatch")
                        if (replay["winner"] != decision["winner"]
                                or replay["actions_sha256"] != decision["actions_sha256"]
                                or replay["retained"] != decision["retained"]):
                            raise ValueError("selection mismatch")
                        if boundary.receipts[0]["inputs"]["pixels"]["sha256"] != \
                                decision["receipts"][0]["inputs"]["pixels"]["sha256"]:
                            raise ValueError("RGB observation mismatch")
                        n = decision["dispatched"]
                        for index in range(n):
                            _, _, term, trunc, info = actor.step(replay["actions"][index].copy())
                            expected_step = episode["trace"][cursor + index]
                            if (bool(term) != expected_step["terminated"] or bool(trunc) != expected_step["truncated"]
                                    or abs(float(info["distance_to_target"]) - expected_step["distance"]) > 1e-6):
                                raise ValueError("physical trajectory mismatch")
                        cursor += n
                        old_tail = replay["actions"][n:].copy()
                        checked += 1
                    if cursor != len(episode["trace"]):
                        raise ValueError("trace length")
                print(f"root={root} replayed", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    result = {"input_sha256": file_hash(run.OUT), "runner_sha256": file_hash(__file__),
              "model_sha256": fingerprint, "checked_decisions": checked, "max_score_abs_error": max_error,
              "scope": "same-model same-environment replay, not independent confirmation"}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
