"""Same-backend replay of sealed actions; no planner or new task search."""
import json
import traceback

import numpy as np

from experiments import pc_v22_timed_feedback_reacher_wiring_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT
    output = out / "physical_replay.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "context_audit.json").read_text())
    if audit["status"] != "passed" or audit["first_audit_sha256"] != seal.digest(out / "audit.json"):
        raise ValueError("source audit mismatch")
    seal.verify(manifest["dependencies"])
    source.preflight.reacher_render_probe()
    import gymnasium as gym

    old, setup = source.old, source.old.setup
    setup.install_missing_field_compat()
    model_source = json.loads(old.run.SOURCE.read_text())
    stats = source.Path(model_source["action_stats_path"])
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                        target_delta=(.45, -.50))
    rows = []
    try:
        for arm in source.ARMS:
            execution = json.loads((out / f"{arm}_execution.json").read_text())
            try:
                if execution["status"] != "completed":
                    raise ValueError("source arm did not complete")
                packets, _ = setup.sample_history_packets(histories, seed=source.TASK_SEED)
                context = setup.collect_executed_context(env, seed=source.TASK_SEED, history_len=3, action_block=5,
                    warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                distances = [old.physical(env)["distance"]]
                for step, event in enumerate(execution["events"]):
                    with np.load(out / f"{arm}_context{step}.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["frames"], np.stack(context["frames"]))
                        np.testing.assert_array_equal(saved["root"], context["root_frame"])
                        np.testing.assert_array_equal(saved["goal"], context["goal"])
                        np.testing.assert_array_equal(saved["history"], context["history_action_blocks"])
                    context = old.context_lib.advance_executed_context(env, context=context,
                        raw_action_block=np.asarray(event["dispatch"], np.float32),
                        action_block=5, verified_root_frame=context["root_frame"])
                    distances.append(old.physical(env)["distance"])
                with np.load(out / f"{arm}_context3.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["frames"], np.stack(context["frames"]))
                    np.testing.assert_array_equal(saved["root"], context["root_frame"])
                    np.testing.assert_array_equal(saved["history"], context["history_action_blocks"])
                np.testing.assert_allclose(distances, [row["distance"] for row in execution["physical_evaluator"]],
                                           rtol=0, atol=1e-9)
                rows.append({"arm": arm, "status": "passed", "distances": distances})
            except Exception:
                rows.append({"arm": arm, "status": "failed", "error": traceback.format_exc()})
    finally:
        env.close()
    seal.verify(manifest["dependencies"])
    old.host.write_new(output, {"rows": rows, "context_audit_sha256": seal.digest(out / "context_audit.json"),
                               "scope": "same-backend replay; not independent replication"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
