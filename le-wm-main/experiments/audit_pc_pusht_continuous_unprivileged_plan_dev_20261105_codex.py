"""Independently verify sealed decisions and replay their physical actions."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_continuous_unprivileged_plan_dev_20261105_codex as run


def main():
    run.verify_prepared()
    out, root, src = run.OUT, run.ROOT, run.src
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    overlay = (root / ".jepa-physics-site").resolve()
    assert os.environ.get("PYTHONPATH") == str(overlay)
    import gym
    import pymunk
    import skimage
    assert (gym.__version__, pymunk.version, skimage.__version__) == ("0.23.1", "6.8.0", "0.25.2")
    assert all(Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))
    base = src.visible.core.old.old.prior.helpers(root)
    runtime, parent = base.load_runtime(root)
    assert runtime.probe(root, parent, root / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] == \
        "native_import_ready_physics_not_executed"
    spec = importlib.util.spec_from_file_location("unprivileged_audit_native", root / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                          for name in ("states.pth", "velocities.pth")]
    report = json.loads((out / "report.json").read_text())
    seal = json.loads((out / "plan_seal.json").read_text())
    assert report["plan_seal_sha256"] == run.sha(out / "plan_seal.json")
    assert report["env_sha256"] == run.sha(root / runtime.ENV_REL)
    assert seal["prepared_sha256"] == run.sha(out / "prepared.json")
    audit_rows = []
    for row in report["rows"]:
        seq, arm = row["sequence"], row["arm"]
        path, source = out / f"root_{seq}_{arm}", src.OUT / f"root_{seq}"
        decision_path = path / "output/decision.json"
        decision = json.loads(decision_path.read_text())
        assert run.sha(decision_path) == row["decision_sha256"] == seal["decisions"][f"{seq}_{arm}"]
        assert decision["worker_sha256"] == run.sha(run.WORKER)
        for key, filename in (("input_sha256", "context.npz"), ("config_sha256", "config.json"),
                              ("metadata_sha256", "model_metadata.json")):
            assert decision[key] == run.sha(path / "input" / filename)
        selected = np.asarray(decision["decision"]["selected"]["chain"], dtype=np.float32)
        assert selected.shape == (15, 2) and np.isfinite(selected).all() and np.max(np.abs(selected)) <= 1
        calls, evaluations = decision["calls"], decision["evaluations"]
        assert len(calls) == len(evaluations)
        token_sum = 0
        scores_full = []
        for call, trace in zip(calls, evaluations, strict=True):
            actions, scores = np.asarray(trace["actions"], np.float32), np.asarray(trace["scores"])
            assert call["status"] == "completed" and actions.shape == (call["samples"], call["physical_steps"], 2)
            assert scores.shape == (len(actions),)
            assert call["prediction_tokens"] == len(actions) * actions.shape[1] // 5
            token_sum += call["prediction_tokens"]
            if actions.shape[1] == 15:
                scores_full.extend(scores.tolist())
        assert token_sum == row["prediction_tokens"] == decision["resource"]["prediction_tokens"] == 1920
        assert np.isclose(row["predicted_score"], min(scores_full), atol=1e-7, rtol=1e-5)
        assert run.sha(path / "physical.npz") == row["physical_sha256"]
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            recorded_states, recorded_images, actions, target = (f[k].copy() for k in ("states", "images", "actions", "target"))
        np.testing.assert_array_equal(actions, selected)
        with np.load(source / "physical.npz", allow_pickle=False) as f:
            warm, expected_history = f["warm"].copy(), f["history_states"].copy()
            np.testing.assert_array_equal(f["goal_state"], target)
        with np.load(path / "input/context.npz", allow_pickle=False) as f:
            rgb_history = f["rgb_history"].copy()
        initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
        env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial))
        try:
            env.seed(0)
            obs, state = env.reset()
            np.testing.assert_array_equal(state, expected_history[0])
            np.testing.assert_array_equal(obs["visual"], rgb_history[0])
            for t, action in enumerate(warm):
                obs, _, done, info = env.step(action.astype(np.float64))
                assert not done
                if t in (4, 9):
                    h = 1 if t == 4 else 2
                    np.testing.assert_array_equal(info["state"], expected_history[h])
                    np.testing.assert_array_equal(obs["visual"], rgb_history[h])
            np.testing.assert_array_equal(info["state"], recorded_states[0])
            np.testing.assert_array_equal(obs["visual"], recorded_images[0])
            for j, action in enumerate(actions):
                obs, _, done, info = env.step(action.astype(np.float64))
                assert not done
                np.testing.assert_array_equal(info["state"], recorded_states[j+1])
                np.testing.assert_array_equal(obs["visual"], recorded_images[j+1])
        finally:
            env.close()
        end = recorded_states[-1]
        joint = float(np.linalg.norm(end[:4] - target[:4]))
        obj = float(np.linalg.norm(end[2:4] - target[2:4]))
        angle = float(abs((end[4] - target[4] + np.pi) % (2*np.pi) - np.pi))
        assert np.isclose(joint, row["joint_error"]) and np.isclose(obj, row["object_error"])
        assert np.isclose(angle, row["angle_error"]) and bool(joint < 20 and angle < np.pi/9) == row["success"]
        assert row["center_outside_frames"] == int(np.sum(np.any((recorded_states[:, :4] < 40) | (recorded_states[:, :4] > 472), axis=1)))
        audit_rows.append({"sequence": seq, "arm": arm, "frames_replayed": 15,
                           "prediction_tokens_rebuilt": token_sum, "selected_is_min_scored_full_chain": True})
        print(f"sequence {seq} {arm}: accounting, selection and physical replay checked", flush=True)
    assert [(x["sequence"], x["arm"]) for x in audit_rows] == [(9, "response"), (9, "cem"), (11, "response"), (11, "cem")]
    run.verify_prepared()
    with (out / "audit.json").open("x") as f:
        json.dump({"status": "passed", "auditor_sha256": run.sha(Path(__file__)),
                   "report_sha256": run.sha(out / "report.json"), "rows": audit_rows,
                   "scope": "saved trace and same-backend physical replay, no independent neural or isolation rerun"},
                  f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
