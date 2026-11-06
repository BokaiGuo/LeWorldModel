"""Independently reconstruct the bank and replay the same physics backend."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_jepa_visible_local_bank_dev_20261105_codex as run
from experiments.audit_pc_jepa_multiroot_action_effect_dev_20261105_codex import order


def main():
    run.verify()
    core, out = run.core, run.OUT
    sha = core.sha
    report = json.loads((out / "report.json").read_text())
    assert report["manifest_sha256"] == sha(out / "local_manifest.json")
    overlay = (core.ROOT / ".jepa-physics-site").resolve()
    assert os.environ.get("PYTHONPATH") == str(overlay)
    import gym
    import pymunk
    import skimage
    assert (gym.__version__, pymunk.version, skimage.__version__) == ("0.23.1", "6.8.0", "0.25.2")
    assert all(Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))
    base = core.old.old.prior.helpers(core.ROOT)
    runtime, parent = base.load_runtime(core.ROOT)
    assert runtime.probe(core.ROOT, parent, core.ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] == \
        "native_import_ready_physics_not_executed"
    spec = importlib.util.spec_from_file_location("visible_audit_native", core.ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    rng = np.random.default_rng(2026110620)
    reference = rng.uniform(-.12, .12, (3, 2))
    near = np.clip(reference + rng.uniform(-.01, .01, (32, 3, 2)), -1, 1)
    far = rng.uniform(-.2, .2, (32, 3, 2))
    bank = np.repeat(np.concatenate((near, far, reference[None])).astype(np.float32), 5, axis=1)
    assert len({x.tobytes() for x in bank}) == 65
    np.testing.assert_array_equal(bank[64], np.load(out / "evaluation_reference.npy", allow_pickle=False))
    warm = np.asarray(core.old.old.prior.SCHEDULE, np.float64).reshape(15, 2)[:10]
    rows = []
    for i, seed in enumerate((2026110621, 2026110622, 2026110623, 2026110624)):
        path = out / f"root_{i}"
        np.testing.assert_array_equal(bank[:64], np.load(path / "input/bank.npy", allow_pickle=False))
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target, history = (f[k].copy() for k in ("states", "goal_state", "history_states"))
        with np.load(path / "input/images.npz", allow_pickle=False) as f:
            frames, goal_rgb = f["terminal_rgb"].copy(), f["goal_rgb"].copy()
        with np.load(path / "input/context.npz", allow_pickle=False) as f:
            images = f["rgb_history"].copy()
            np.testing.assert_array_equal(f["goal_rgb"][0], goal_rgb)
        for j, chain in enumerate(bank):
            env = native.PushTEnv(**base.CTOR)
            try:
                env.seed(seed)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], images[0])
                for step, action in enumerate(warm):
                    obs, _, done, info = env.step(action.copy())
                    assert not done
                    if step in (4, 9):
                        h = 1 if step == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], images[h])
                for action in chain.astype(np.float64):
                    obs, _, done, info = env.step(action)
                    assert not done
                np.testing.assert_array_equal(info["state"], target if j == 64 else states[j])
                np.testing.assert_array_equal(obs["visual"], goal_rgb if j == 64 else frames[j])
            finally:
                env.close()
        receipt = json.loads((path / "output/receipt.json").read_text())
        assert receipt["status"] == "completed"
        assert receipt["worker_sha256"] == sha(core.old.WORKER)
        assert receipt["latents_sha256"] == sha(path / "output/latents.npz")
        for key, filename in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                              ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            assert receipt[key] == sha(path / "input" / filename)
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        scores = {"predicted": ((pred-goal)**2).mean(axis=(1, 2)),
                  "actual_encoded": ((actual-goal)**2).mean(axis=(1, 2))}
        pos = np.linalg.norm(states[:, :4] - target[:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2*np.pi) - np.pi)
        success = (pos < 20) & (angle < np.pi/9)
        r = report["rows"][i]
        assert success.tolist() == r["success_by_index"]
        assert r["goal_centers_qualified"] == bool(np.all((target[:4] >= 40) & (target[:4] <= 472)))
        assert r["candidate_center_outside"] == int(np.sum(np.any((states[:, :4] < 40) | (states[:, :4] > 472), axis=1)))
        assert np.isclose(r["goal_object_displacement"], np.linalg.norm(target[2:4] - history[-1, 2:4]))
        assert np.isclose(r["mean_prediction_mse"], np.mean((pred-actual)**2))
        assert order(scores["predicted"], scores["actual_encoded"]) == tuple(
            r["predicted_vs_actual"][k] for k in ("agree", "disagree", "tied"))
        for group, indices in (("all", np.arange(64)), ("near", np.arange(32)), ("far", np.arange(32, 64))):
            assert r["groups"][group]["success_count"] == int(success[indices].sum())
            for name, values in scores.items():
                ranked = indices[np.argsort(values[indices], kind="stable")]
                hits = np.flatnonzero(success[ranked])
                row = r["groups"][group][name]
                assert row["index"] == int(ranked[0]) and row["success"] == bool(success[ranked[0]])
                assert row["first_success_rank"] == (int(hits[0]) + 1 if len(hits) else None)
                assert np.isclose(row["score"], values[ranked[0]])
        rows.append({"root": i, "replayed": 65, "goal_centers_qualified": r["goal_centers_qualified"],
                     "goal_object_displacement": r["goal_object_displacement"]})
        print(f"root {i}: 65 physical chains replayed and scores checked", flush=True)
    run.verify()
    output = {"status": "passed", "report_sha256": sha(out / "report.json"),
              "auditor_sha256": sha(Path(__file__)), "rows": rows,
              "scope": "same physics backend, no independent neural rerun"}
    with (out / "audit.json").open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
