"""Independently replay the executed actions of the scored-incumbent arm."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_pusht_explicit_prefix_incumbent_runner_20261107_codex as run
from experiments.pc_input_permissions import file_hash
from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot
from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

OUTPUT = Path(str(run.STEM) + ".replay.json")


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest_path = run.STEM.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for key, path in (("runner_sha256", run.__file__), ("solver_sha256", run.solver.__file__),
                      ("protocol_sha256", run.PROTOCOL), ("trace_sha256", run.STEM.with_suffix(".trace.jsonl")),
                      ("audit_sha256", run.STEM.with_suffix(".audit.jsonl")),
                      ("metrics_sha256", run.STEM / "eval_metrics.json")):
        if manifest[key] != file_hash(path):
            raise ValueError(f"source drift {key}")
    trace = load(run.STEM.with_suffix(".trace.jsonl"))
    rows = []
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        panel = panel_rows(data, seed=85, count=50)
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=100)
        try:
            for index, data_row in enumerate(panel):
                seed = 871001 + index
                commands = trace[seed]
                env.reset(seed=seed)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["state"][data_row], dtype=np.float64))
                actor._set_goal_state(np.asarray(data["state"][data_row + 25], dtype=np.float64))
                first_success = None
                for step, entry in enumerate(commands, 1):
                    if not np.allclose(snapshot(actor), entry["before"], atol=1e-4, rtol=0):
                        raise ValueError(f"before mismatch {index=} {step=}")
                    action = np.asarray(entry["action"], np.float32)
                    if not np.isfinite(action).all() or np.abs(action).max() > 1 + 1e-6:
                        raise ValueError("invalid dispatched action")
                    _, _, term, trunc, _ = actor.step(action.copy())
                    if not np.allclose(snapshot(actor), entry["after"], atol=1e-4, rtol=0):
                        raise ValueError(f"after mismatch {index=} {step=}")
                    if bool(term) != entry["terminated"] or bool(trunc) != entry["truncated"]:
                        raise ValueError("outcome mismatch")
                    if term and first_success is None:
                        first_success = step
                rows.append({"index": index, "data_row": int(data_row), "steps": len(commands),
                             "first_success_step": first_success})
        finally:
            env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"manifest_sha256": file_hash(manifest_path), "runner_sha256": file_hash(__file__),
                   "scope": "same simulator executed-action replay; model choices not rescored", "rows": rows},
                  stream, indent=2)
    print(f"replayed {len(rows)} tasks and {sum(row['steps'] for row in rows)} steps", flush=True)


if __name__ == "__main__":
    main()
