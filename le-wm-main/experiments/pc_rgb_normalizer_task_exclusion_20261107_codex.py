"""Audit task membership in scaler fit and measure leave-panel-out feedback sensitivity."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as source
from experiments.pc_input_permissions import RGB, RGBLeWM, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_NORMALIZER_TASK_EXCLUSION_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_normalizer_task_exclusion_20261107_codex.json"


def fit_scalers(actions, episode_ids, excluded):
    actions, episode_ids = np.asarray(actions), np.asarray(episode_ids)
    excluded = np.asarray(excluded)
    if actions.ndim != 2 or actions.shape[1] != 2 or episode_ids.shape != (len(actions),):
        raise ValueError("invalid action or episode arrays")
    valid = ~np.isnan(actions).any(axis=1)
    if not np.isfinite(actions[valid]).all():
        raise ValueError("nonfinite action")
    excluded_mask = np.isin(episode_ids, excluded) & valid
    if not excluded_mask.any() or not (valid & ~excluded_mask).any():
        raise ValueError("empty excluded or retained sample")
    full = StandardScaler().fit(actions[valid])
    reduced = StandardScaler().fit(actions[valid & ~excluded_mask])
    counts = {int(ep): int(np.sum(excluded_mask & (episode_ids == ep))) for ep in excluded}
    return full, reduced, counts


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUT.exists():
        raise FileExistsError(OUT)
    history = json.loads(source.OUTPUT.read_text())
    if (history["runner_sha256"] != file_hash(source.__file__)
            or history["protocol_sha256"] != file_hash(source.PROTOCOL)
            or history["input_sha256"] != file_hash(source.INPUT)):
        raise ValueError("history source drift")
    model, full_reference, transform, normalizer = source.common.model_setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != history["model_sha256"] or normalizer != history["normalizer_sha256"]:
        raise ValueError("historical model or normalizer drift")
    with h5py.File(source.common.DATA, "r") as h5:
        actions = np.asarray(h5["action"][:])
        eps = np.asarray(h5["ep_idx"][:])
    valid = ~np.isnan(actions).any(axis=1)
    if hashlib.sha256(np.ascontiguousarray(actions[valid]).tobytes()).hexdigest() != normalizer:
        raise ValueError("historical fit rows changed")
    full, reduced, counts = fit_scalers(actions, eps, np.arange(9000, 9040))
    np.testing.assert_array_equal(full.mean_, full_reference.mean_)
    np.testing.assert_array_equal(full.scale_, full_reference.scale_)
    if any(count == 0 for count in counts.values()):
        raise ValueError("at least one panel task has no fitted rows")
    audit = freeze_audit(RGB, "lewm", [Path(__file__), PROTOCOL, source.INPUT,
                         source.common.CHECKPOINT / "weights.pt", source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; panel-excluded action scaler",
                         clock="five environment steps per packet")
    boundary = RGBLeWM(model, audit)
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Test 40 excluded-panel action-normalizer tasks with actual RGB feedback.", flush=True)
    try:
        with np.load(source.INPUT, allow_pickle=False) as inputs, \
                h5py.File(source.common.DATA, "r") as data, torch.inference_mode():
            for index, historic in enumerate(history["rows"]):
                ep, row = historic["episode"], historic["row"]
                if ep != 9000 + index or int(inputs["rows"][index]) != row:
                    raise ValueError("task binding mismatch")
                goal = transform(inputs["goal"][index])[None, None, None].cuda()
                arms = {}
                for method in ("effect", "cem16"):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row])
                    actor._set_goal_state(data["proprio"][row+25])
                    tail, trace, decisions = None, [], []
                    for iteration in range(5):
                        remaining = 25 - len(trace)
                        current = {"pixels": transform(actor.render().copy())[None, None, None].cuda(),
                                   "goal": goal, "action": torch.empty((1, 1, 0, 10), device="cuda")}
                        if method == "effect":
                            result = source.program.compile_rgb(boundary, current, scaler=reduced, old_tail=tail,
                                candidate_seed=27121000 + ep * 10 + iteration,
                                strategy_seed=27122000 + ep * 10 + iteration,
                                strategy="effect", remaining_steps=remaining, final_rule="temporal")
                            choice = {"source": result["choice"].source, "index": result["choice"].index}
                            queries, packets = result["model_queries"], result["model_packet_steps"]
                        else:
                            result = source.cem_temporal(boundary, current, reduced, tail, remaining,
                                                         8140000 + ep * 10 + iteration, 16)
                            choice = {"index": result["chosen"]}
                            queries, packets = result["queries"], result["packet_steps"]
                        chosen = result["actions"]
                        dispatched = []
                        for action in chosen[:5]:
                            _, _, term, trunc, info = actor.step(action.copy())
                            dispatched.append(action.tolist())
                            trace.append({"action": action.tolist(), "distance": float(info["distance_to_target"]),
                                          "terminated": bool(term), "truncated": bool(trunc)})
                            if term or trunc:
                                break
                        decisions.append({"choice": choice, "remaining": remaining, "queries": queries,
                                          "packet_steps": packets, "actions": chosen.tolist(), "dispatched": dispatched})
                        tail = chosen[len(dispatched):].copy()
                        if term or trunc:
                            break
                    previous = historic["arms"][method]
                    arms[method] = {"trace": trace, "decisions": decisions,
                                    "success": any(x["terminated"] for x in trace),
                                    "historical_success": previous["ever_success"]}
                rows.append({"episode": ep, "row": row, "excluded_fit_rows": counts[ep], "arms": arms})
                print(f"episode={ep} effect={arms['effect']['success']} cem={arms['cem16']['success']}", flush=True)
    finally:
        env.close()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with OUT.open("x") as stream:
        json.dump({"scope": "exposed development only; fitted-row exclusion is not training independence",
                   "track": RGB, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "reference_sha256": file_hash(source.OUTPUT), "model_sha256": fingerprint,
                   "normalizer_action_sha256": normalizer,
                   "normalizer": {"full_mean": full.mean_.tolist(), "full_scale": full.scale_.tolist(),
                                  "excluded_mean": reduced.mean_.tolist(), "excluded_scale": reduced.scale_.tolist(),
                                  "full_count": int(full.n_samples_seen_), "excluded_count": int(reduced.n_samples_seen_),
                                  "panel_rows": int(sum(counts.values()))}, "rows": rows}, stream, separators=(",", ":"),
                  allow_nan=False)


if __name__ == "__main__":
    main()
