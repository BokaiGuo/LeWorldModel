"""Compare full shared-prefix pool with real three-frame context."""
import hashlib
import json
import time
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
from experiments.pc_pusht_all_shared_prefix_rescore_20261107_codex import BASE, OUT as PRIOR, SOURCE
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_pusht_shared_prefix_rescore_20261107_codex import conditional_rollout
from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT, records
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

OUT = ROOT / "results/pc_pusht_shared_prefix_three_frame_20261107_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_SHARED_PREFIX_THREE_FRAME_PROTOCOL_20261107_CODEX.md"


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    previous = json.loads(SOURCE.with_suffix(".json").read_text())
    sealed = json.loads(PRIOR.with_suffix(".json").read_text())
    if file_hash(SOURCE.with_suffix(".npz")) != previous["arrays_sha256"]:
        raise ValueError("alignment arrays changed")
    if file_hash(PRIOR.with_suffix(".npz")) != sealed["arrays_sha256"]:
        raise ValueError("single-frame arrays changed")
    trace = records(BASE / "trace.jsonl")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model_sha256"] or fingerprint != sealed["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    scores, plans = {}, []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as arrays, h5py.File(DATA, "r") as data:
        action_data = data["action"][:]
        scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            with torch.inference_mode():
                for plan in sealed["plans"]:
                    task = plan["task"]
                    index = next(i for i, x in enumerate(previous["rows"]) if x["task"] == task and x["selected"])
                    batches = [torch.load(BASE / "capture" / f"call_{task * 5 + c:03}.pt",
                                          map_location="cpu", weights_only=True) for c in range(5)]
                    chains = []
                    for item in plan["candidates"]:
                        action = batches[item["call"]]["actions"][0, item["slot"]].numpy()
                        if hashlib.sha256(action.tobytes()).hexdigest() != item["sha256"]:
                            raise ValueError("candidate identity mismatch")
                        chains.append(action)
                    chain = np.stack(chains)
                    selected = chain[plan["original_selected_position"]]
                    if not np.all(chain[:, :2] == selected[None, :2]):
                        raise ValueError("nonshared action prefix")
                    row = int(next(x for x in previous["rows"] if x["task"] == task and x["selected"])["index"])
                    source_row = int(panel_rows(data, seed=85, count=4)[task])
                    env.reset(seed=871001 + task)
                    actor = env.unwrapped
                    actor._set_state(np.asarray(data["state"][source_row], np.float64))
                    actor._set_goal_state(np.asarray(data["state"][source_row + 25], np.float64))
                    task_trace = sorted((x for x in trace if x["reset_seed"] == 871001 + task), key=lambda x: x["step"])
                    np.testing.assert_allclose(snapshot(actor), task_trace[0]["before"], atol=1e-4, rtol=0)
                    frames = [transform(native_world_pixels(actor.render()))]
                    physical_actions = scaler.inverse_transform(selected.reshape(25, 2))
                    for t, action in enumerate(physical_actions[:10], 1):
                        np.testing.assert_allclose(action, task_trace[t - 1]["action"], atol=1e-6, rtol=0)
                        actor.step(action.copy())
                        np.testing.assert_allclose(snapshot(actor), task_trace[t - 1]["after"], atol=1e-4, rtol=0)
                        if t % 5 == 0:
                            frames.append(transform(native_world_pixels(actor.render())))
                    history = model.encode({"pixels": torch.stack(frames).to("cuda")[:, None]})["emb"][:, 0]
                    torch.testing.assert_close(history[1:], torch.as_tensor(arrays["actual"][index, :2], device="cuda"),
                                               atol=1e-5, rtol=1e-5)
                    goal = torch.as_tensor(arrays["goal"][index], device="cuda")
                    actions = torch.as_tensor(chain, device="cuda")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    pred = conditional_rollout(model, history, actions, 2)
                    values = (pred[:, -1] - goal).square().sum(-1)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    values = values.cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite scores")
                    winner = int(np.argmin(values))
                    scores[f"task_{task}"] = values
                    plans.append({"task": task, "selected_physical_index": row, "candidate_count": len(chain),
                                  "winner": winner, "winner_sha256": plan["candidates"][winner]["sha256"],
                                  "predictor_seconds": elapsed, "model_packet_evaluations": 3 * len(chain),
                                  "predict_calls": 3})
                    print(json.dumps(plans[-1]), flush=True)
        finally:
            env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **scores)
    with OUT.with_suffix(".json").open("x") as stream:
        json.dump({"model_sha256": fingerprint, "protocol_sha256": file_hash(PROTOCOL),
                   "runner_sha256": file_hash(Path(__file__)), "source_sha256": file_hash(SOURCE.with_suffix(".json")),
                   "prior_sha256": file_hash(PRIOR.with_suffix(".json")),
                   "arrays_sha256": file_hash(OUT.with_suffix(".npz")), "plans": plans}, stream, indent=2)


if __name__ == "__main__":
    main()
