"""Same-backend physical replay of sealed Reacher dispatches."""
import json
import traceback

import numpy as np

from experiments import pc_v17_reacher_score_response_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    output = source.OUT / "physical_replay.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    summary = json.loads((source.OUT / "summary.json").read_text())
    seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or audit["summary_sha256"] != seal.digest(source.OUT / "summary.json")
            or summary["manifest_sha256"] != seal.digest(source.OUT / "manifest.json")):
        raise ValueError("source audit mismatch")
    source.preflight.reacher_render_probe()
    import gymnasium as gym

    setup, old = source.old.setup, source.old
    setup.install_missing_field_compat()
    model_source = json.loads(old.run.SOURCE.read_text())
    stats = source.Path(model_source["action_stats_path"])
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                        target_delta=(.45, -.50))
    rows = []
    try:
        for arm in source.ARMS:
            execution = json.loads((source.OUT / f"{arm}_execution.json").read_text())
            if execution["status"] != "completed":
                rows.append({"arm": arm, "status": "source_failed", "error": execution["error"]})
                continue
            try:
                packets, _ = setup.sample_history_packets(histories, seed=source.TASK_SEED)
                context = setup.collect_executed_context(env, seed=source.TASK_SEED, history_len=3, action_block=5,
                    warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                distances = [old.physical(env)["distance"]]
                for step, event in enumerate(execution["events"]):
                    with np.load(source.OUT / f"{arm}_context{step}.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["frames"], np.stack(context["frames"]))
                        np.testing.assert_array_equal(saved["root"], context["root_frame"])
                        np.testing.assert_array_equal(saved["goal"], context["goal"])
                        np.testing.assert_array_equal(saved["history"], context["history_action_blocks"])
                    context = old.context_lib.advance_executed_context(env, context=context,
                        raw_action_block=np.asarray(event["dispatch"], np.float32),
                        action_block=5, verified_root_frame=context["root_frame"])
                    distances.append(old.physical(env)["distance"])
                with np.load(source.OUT / f"{arm}_context3.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["frames"], np.stack(context["frames"]))
                    np.testing.assert_array_equal(saved["root"], context["root_frame"])
                    np.testing.assert_array_equal(saved["history"], context["history_action_blocks"])
                expected = [x["distance"] for x in execution["physical_evaluator"]]
                np.testing.assert_allclose(distances, expected, rtol=0, atol=1e-9)
                rows.append({"arm": arm, "status": "passed", "distances": distances})
            except Exception:
                rows.append({"arm": arm, "status": "failed", "error": traceback.format_exc()})
    finally:
        env.close()
    seal.verify(manifest["dependencies"])
    old.host.write_new(output, {"rows": rows, "audit_sha256": seal.digest(source.OUT / "audit.json"),
        "scope": "same-backend replay; not independent replication"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
