"""Replay recorded observations/actions and recompute every planning decision on GPU."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as run
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

OUTPUT = run.ROOT / "results/pc_tworoom_temporal_new_episodes_gpu_replay_20261107_codex.json"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = json.loads(run.OUTPUT.read_text())
    if (original["runner_sha256"] != file_hash(run.__file__)
            or original["input_sha256"] != file_hash(run.INPUT)
            or original["protocol_sha256"] != file_hash(run.PROTOCOL)):
        raise ValueError("sealed input or runner drift")
    model, scaler, transform, normalizer = run.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model_sha256"] or normalizer != original["normalizer_sha256"]:
        raise ValueError("model or normalizer drift")
    audit = freeze_audit(RGB, "lewm", [Path(run.__file__), run.PROTOCOL, run.INPUT, run.REFERENCE,
                                      Path(run.program.__file__), run.common.CHECKPOINT / "weights.pt",
                                      run.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224 TwoRoom dataset action scaler",
                         clock="5-step packets and remaining 25-step deadline")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    records = []
    try:
        with (np.load(run.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.common.DATA, "r") as source, torch.inference_mode()):
            for i, row in enumerate(original["rows"]):
                if row["episode"] != int(inputs["episodes"][i]) or row["row"] != int(inputs["rows"][i]):
                    raise ValueError("input identity mismatch")
                goal = transform(inputs["goal"][i])[None, None, None].cuda()
                for arm in run.ARMS:
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(source["proprio"][row["row"]])
                    actor._set_goal_state(source["proprio"][row["row"] + 25])
                    old_tail = None
                    offset = 0
                    for dec in row["arms"][arm]["decisions"]:
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        boundary = RGBLeWM(model, audit)
                        remaining = dec["remaining"]
                        if arm in ("lookahead", "effect"):
                            replay = run.program.compile_rgb(
                                boundary, current, scaler=scaler, old_tail=old_tail,
                                candidate_seed=dec["candidate_seed"], strategy_seed=dec["strategy_seed"],
                                strategy=arm, remaining_steps=remaining, final_rule="temporal")
                            recorded_costs = np.asarray(sum(dec["temporal_scores"], []))
                            replay_costs = np.asarray(sum(replay["temporal_scores"], []))
                            recorded_choice = [dec["choice"]["source"], dec["choice"]["index"]]
                            replay_choice = [replay["choice"].source, replay["choice"].index]
                            same_retained = dec["retained"] == replay["retained_prefixes"]
                        else:
                            replay = run.cem_temporal(boundary, current, scaler, old_tail, remaining,
                                                      dec["candidate_seed"], 16 if arm == "cem16" else 300)
                            recorded_costs = np.asarray([dec["incumbent_cost"]] +
                                                        sum((r["costs"] for r in dec["rounds"]), []))
                            replay_costs = np.asarray([replay["incumbent_cost"]] +
                                                      sum((r["costs"] for r in replay["rounds"]), []))
                            recorded_choice, replay_choice = dec["chosen"], replay["chosen"]
                            same_retained = None
                        original_actions = np.asarray(dec["actions"], dtype=np.float32)
                        action_error = float(np.max(np.abs(original_actions - replay["actions"])))
                        error = float(np.max(np.abs(recorded_costs - replay_costs)))
                        physical_error = 0.0
                        for action in dec["dispatched"]:
                            _, _, term, trunc, info = actor.step(np.asarray(action, dtype=np.float32))
                            prev = row["arms"][arm]["trace"][offset]
                            physical_error = max(physical_error, abs(float(info["distance_to_target"]) - prev["distance"]))
                            if bool(term) != prev["terminated"] or bool(trunc) != prev["truncated"]:
                                raise ValueError("physical outcome mismatch")
                            offset += 1
                        records.append({"episode": row["episode"], "arm": arm, "iteration": dec["iteration"],
                                        "cost_max_abs_error": error, "action_max_abs_error": action_error,
                                        "physical_distance_max_abs_error": physical_error,
                                        "same_choice": recorded_choice == replay_choice,
                                        "same_retained": same_retained})
                        old_tail = original_actions[len(dec["dispatched"]):].copy()
                print(f"replayed episode {row['episode']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(run.OUTPUT), "input_sha256": file_hash(run.INPUT),
                   "runner_sha256": file_hash(run.__file__), "audit_sha256": file_hash(__file__),
                   "model_sha256": fingerprint, "records": records}, stream, allow_nan=False)
    print(f"{len(records)} decisions; max score error={max(r['cost_max_abs_error'] for r in records):.9g}; "
          f"max action error={max(r['action_max_abs_error'] for r in records):.9g}; "
          f"choice changes={sum(not r['same_choice'] for r in records)}")


if __name__ == "__main__":
    main()
