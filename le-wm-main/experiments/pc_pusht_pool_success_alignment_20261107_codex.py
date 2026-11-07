"""Post-hoc model/real-image alignment of selected and successful pool chains."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from eval import img_transform
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint, native_world_pixels
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_pusht_scored_first_pool_run_20261107_codex import BASE
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT, records
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

OUT = ROOT / "results/pc_pusht_pool_success_alignment_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_POOL_SUCCESS_ALIGNMENT_PROTOCOL_20261107_CODEX.md"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    traces = records(BASE / "trace.jsonl")
    all_pred, all_real, all_goal, rows = [], [], [], []
    print("Align selected and all previously successful candidate chains; evaluator truth is not used by planner.", flush=True)
    with h5py.File(DATA, "r") as data, torch.inference_mode():
        panel = panel_rows(data, seed=85, count=4)
        source_actions = data["action"][:]
        scaler = StandardScaler().fit(source_actions[~np.isnan(source_actions).any(axis=1)])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for task, raw_row in enumerate(panel):
                row = int(raw_row)
                physical = json.loads((BASE / f"physical_{task}.json").read_text())
                if row != physical["data_row"]:
                    raise ValueError("task identity mismatch")
                selected = physical["selected_index"]
                chosen = [item for item in physical["records"]
                          if item["first_success_step"] is not None or item["index"] == selected]
                goal = transform(np.asarray(data["pixels"][row + 25]))[None, None].to("cuda")
                encoded_goal = model.encode({"pixels": goal})["emb"][0, 0]
                task_trace = [entry for entry in traces if entry["reset_seed"] == 871001 + task]
                if not task_trace:
                    raise ValueError("missing initial trace")
                for item in chosen:
                    occurrence = item["occurrences"][0]
                    path = BASE / "capture" / f"call_{task * 5 + occurrence['call']:03}.pt"
                    batch = torch.load(path, map_location="cpu", weights_only=True)
                    slot = occurrence["slot"]
                    action = batch["actions"][0, slot].numpy()
                    if hashlib.sha256(action.tobytes()).hexdigest() != item["action_sha256"]:
                        raise ValueError("candidate identity mismatch")
                    predicted = batch["prediction"][0, slot, 1:].numpy()
                    query_goal = batch["goal"][0, -1].numpy()
                    np.testing.assert_allclose(encoded_goal.cpu().numpy(), query_goal, atol=1e-4, rtol=1e-5)
                    physical_actions = scaler.inverse_transform(action.reshape(25, 2))
                    if not np.isfinite(physical_actions).all() or np.abs(physical_actions).max() > 1:
                        raise ValueError("illegal action")
                    env.reset(seed=871001 + task)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][row], np.float64))
                    actor._set_goal_state(np.asarray(data["state"][row + 25], np.float64))
                    np.testing.assert_allclose(snapshot(actor), task_trace[0]["before"], atol=1e-4, rtol=0)
                    frames, successes = [], []
                    for t, physical_action in enumerate(physical_actions, 1):
                        obs, _, term, trunc, _ = actor.step(physical_action.copy())
                        if trunc:
                            raise ValueError("unexpected truncation")
                        successes.append(bool(term))
                        if t % 5 == 0:
                            np.testing.assert_allclose(obs["state"], item["packet_states"][t // 5 - 1],
                                                       atol=1e-4, rtol=0)
                            frames.append(transform(native_world_pixels(actor.render())))
                    first = next((i + 1 for i, value in enumerate(successes) if value), None)
                    if first != item["first_success_step"] or successes[-1] != item["final_success"]:
                        raise ValueError("physical success mismatch")
                    actual = model.encode({"pixels": torch.stack(frames).to("cuda")[:, None]})["emb"][:, 0]
                    all_pred.append(predicted)
                    all_real.append(actual.cpu().numpy())
                    all_goal.append(query_goal)
                    rows.append({"task": task, "index": item["index"], "selected": item["index"] == selected,
                                 "successes": successes, "recorded_cost": occurrence["cost"],
                                 "action_sha256": item["action_sha256"]})
                print(f"task={task}, aligned={len(chosen)}", flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed during encoding")
    pred, real, goals = map(np.stack, (all_pred, all_real, all_goal))
    pred_cost = ((pred - goals[:, None]) ** 2).sum(-1)
    real_cost = ((real - goals[:, None]) ** 2).sum(-1)
    errors = np.linalg.norm(pred - real, axis=-1)
    for index, row in enumerate(rows):
        np.testing.assert_allclose(pred_cost[index, -1], row["recorded_cost"], atol=0.002, rtol=0.0002)
        row.update(predicted_costs=pred_cost[index].tolist(), actual_encoded_costs=real_cost[index].tolist(),
                   representation_error_l2=errors[index].tolist())
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, predicted=pred, actual=real, goal=goals)
    sources = [PROTOCOL, Path(__file__), BASE / "trace.jsonl"] + [BASE / f"physical_{i}.json" for i in range(4)]
    result = {"model_sha256": fingerprint, "source_sha256": {str(p): file_hash(p) for p in sources},
              "arrays_sha256": file_hash(OUT.with_suffix(".npz")), "rows": rows}
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Saved {len(rows)} aligned chains.", flush=True)


if __name__ == "__main__":
    main()
