"""Paired PushT initialization audit with sealed first plans; no replanning."""
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401
import stable_worldmodel.envs.pusht.env as pusht
from sklearn.preprocessing import StandardScaler

from experiments import pc_pusht_first_plan_alignment_20261107_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import native_world_pixels
from experiments.pc_pusht_goal_render_reindex_20261107_codex import set_render_pose_reindexed
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load

OUTPUT = source.RESULTS / "pc_pusht_reset_tick_same_chain_20261107_codex.json"
PROTOCOL = source.ROOT.parent / "docs/PC_PUSHT_RESET_TICK_SAME_CHAIN_PROTOCOL_20261107_CODEX.md"


def legal_actions(chain, scaler):
    raw = np.asarray(chain, dtype=np.float32)
    if raw.shape != (5, 10) or not np.isfinite(raw).all():
        raise ValueError("invalid model chain")
    actions = scaler.inverse_transform(raw.reshape(25, 2)).astype(np.float32)
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001:
        raise ValueError("illegal physical actions")
    return actions


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    initial = json.loads(source.OUT.with_suffix(".json").read_text())
    if file_hash(source.OUT.with_suffix(".npz")) != initial["arrays_sha256"]:
        raise ValueError("initial alignment data drift")
    for path, sha in initial["source_sha256"].items():
        if file_hash(path) != sha:
            raise ValueError(f"initial alignment source drift: {path}")
    audits = [source.load_first(stem) for stem in (source.OLD, source.NEW)]
    traces = [load(stem.with_suffix(".trace.jsonl")) for stem in (source.OLD, source.NEW)]
    results = []
    print("Pair native and no-tick reset on fixed 50 x 2 first plans; no replanning.", flush=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
    try:
        with h5py.File(source.DATA, "r") as data:
            full_actions = data["action"][:]
            scaler = StandardScaler().fit(full_actions[~np.isnan(full_actions).any(axis=1)])
            for row in initial["rows"]:
                index, data_row = row["index"], row["data_row"]
                state = np.asarray(data["state"][data_row], dtype=np.float64)
                goal = np.asarray(data["state"][data_row + 25], dtype=np.float64)
                data_rgb = np.asarray(data["pixels"][data_row])
                arms = []
                for arm in range(2):
                    if audits[arm][index]["index"] != index:
                        raise ValueError("audit index mismatch")
                    actions = legal_actions(audits[arm][index]["selected_actions"], scaler)
                    modes = {}
                    for mode in ("native", "no_tick"):
                        seed = 871001 + index
                        env.reset(seed=seed)
                        actor = env.unwrapped
                        if mode == "native":
                            actor._set_state(state.copy())
                        else:
                            set_render_pose_reindexed(actor, state)
                        actor._set_goal_state(goal.copy())
                        visible = np.asarray(actor._get_obs())
                        frame = native_world_pixels(actor.render())
                        item = {"visible_initial": visible.tolist(), "initial_snapshot": snapshot(actor).tolist(),
                                "initial_state_max_abs_error": float(np.max(np.abs(visible - state))),
                                "initial_rgb_mean_abs_error": float(np.abs(
                                    frame.astype(np.int16) - data_rgb.astype(np.int16)).mean()),
                                "steps": []}
                        original_trace = traces[arm][seed]
                        for t, action in enumerate(actions):
                            before = snapshot(actor)
                            if mode == "native" and t < len(original_trace):
                                np.testing.assert_allclose(action, original_trace[t]["action"], atol=1e-6, rtol=0)
                                np.testing.assert_allclose(before, original_trace[t]["before"], atol=1e-4, rtol=0)
                            _, _, terminated, truncated, _ = actor.step(action.copy())
                            after = snapshot(actor)
                            if mode == "native" and t < len(original_trace):
                                np.testing.assert_allclose(after, original_trace[t]["after"], atol=1e-4, rtol=0)
                                if bool(terminated) != original_trace[t]["terminated"]:
                                    raise ValueError(f"termination mismatch {index}/{arm}/{t}")
                            item["steps"].append({"action": action.tolist(), "after": after.tolist(),
                                                  "terminated": bool(terminated), "truncated": bool(truncated)})
                        item["ever"] = any(x["terminated"] for x in item["steps"])
                        item["final"] = item["steps"][-1]["terminated"]
                        if mode == "native":
                            expected = row["arms"][arm]
                            if item["ever"] != (expected["first_success"] is not None) or item["final"] != expected["final_success"]:
                                raise ValueError(f"success mismatch {index}/{arm}")
                        modes[mode] = item
                    arms.append(modes)
                results.append({"index": index, "data_row": data_row, "same_chain": row["same_chain"], "arms": arms})
                print(f"paired {index + 1}/50", flush=True)
    finally:
        env.close()
    paths = [PROTOCOL, Path(__file__), Path(pusht.__file__),
             source.OUT.with_suffix(".json"), source.OUT.with_suffix(".npz"),
             source.ROOT / "experiments/pc_pusht_goal_render_reindex_20261107_codex.py"]
    report = {"scope": "exposed development fixed-chain diagnostic, no gate promotion",
              "data_path": str(source.DATA), "data_size": source.DATA.stat().st_size,
              "data_mtime_ns": source.DATA.stat().st_mtime_ns,
              "source_sha256": {str(p): file_hash(p) for p in paths}, "rows": results}
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"completed {len(results)} paired tasks")


if __name__ == "__main__":
    main()
