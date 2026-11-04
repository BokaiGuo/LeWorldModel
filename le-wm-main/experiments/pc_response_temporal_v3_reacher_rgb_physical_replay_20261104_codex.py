"""Replay sealed warmup and executed actions on the same Reacher backend."""
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_reacher_rgb_live_20261104_codex as run

AUDIT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")
OUT = run.PREFIX.with_name(run.PREFIX.name + "_physical_replay.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != run.host.UUID:
        raise ValueError("cooperative GPU UUID required")
    sha = run.host.bridge.sha_file
    manifest, audit = json.loads(run.MANIFEST.read_text()), json.loads(AUDIT.read_text())
    if audit["manifest_sha256"] != sha(run.MANIFEST) or manifest["dependencies"] != run.dependencies(json.loads(run.run.SOURCE.read_text())):
        raise ValueError("source seal mismatch")
    with run.host.lock.gpu_lock(run.host.LOCK):
        import gymnasium as gym
        run.setup.install_missing_field_compat()
        env = run.setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                                target_delta=(.65, -.65))
        findings = []
        try:
            for arm in run.ARMS:
                receipt = json.loads(run.path(arm, "execution.json").read_text())
                with np.load(run.path(arm, "context0.npz"), allow_pickle=False) as context:
                    warmup = context["history"].reshape(-1, 2).astype(np.float32)
                    goal_digest = run.context_lib.rgb_digest(context["goal"])
                    root_digest = run.context_lib.rgb_digest(context["root"])
                _, info = env.reset(seed=20263160)
                if run.context_lib.rgb_digest(info["goal"]) != goal_digest:
                    raise ValueError("replayed goal image mismatch")

                def execute(actions):
                    for action in actions:
                        _, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
                        if terminated or truncated:
                            raise ValueError("replay terminated early")

                execute(warmup)
                observed_root = np.asarray(env.render(), dtype=np.uint8)
                rgb_equal = run.context_lib.rgb_digest(observed_root) == root_digest
                outcomes = [run.physical(env)]
                for item in receipt["rows"]:
                    execute(item["dispatch"])
                    outcomes.append(run.physical(env))
                max_qpos, max_goal = 0., 0.
                for actual, expected in zip(outcomes, receipt["initial_and_block_outcomes"], strict=True):
                    max_qpos = max(max_qpos, float(np.max(np.abs(np.asarray(actual["qpos"]) - expected["qpos"]))))
                    max_goal = max(max_goal, float(np.max(np.abs(np.asarray(actual["target_qpos"]) - expected["target_qpos"]))))
                    if abs(actual["distance"] - expected["distance"]) > 1e-8:
                        raise ValueError("replayed distance differs")
                if max_qpos > 1e-8 or max_goal > 1e-8:
                    raise ValueError("replayed physical state differs")
                findings.append({"arm": arm, "root_rgb_equal": rgb_equal, "max_qpos_abs_error": max_qpos,
                                 "max_target_abs_error": max_goal, "outcomes": outcomes,
                                 "execution_sha256": sha(run.path(arm, "execution.json"))})
        finally:
            env.close()
    run.host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "audit_sha256": sha(AUDIT),
                             "script_sha256": sha(Path(__file__)), "rows": findings,
                             "scope": "same-backend replay; qvel and all intermediate states not checked"})
    print("Both real Reacher action sequences replayed at all four decision boundaries.", flush=True)


if __name__ == "__main__":
    main()
