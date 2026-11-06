"""Independently replay new conditional candidates and check score arithmetic."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_reference_mixed_score_dev_20261105_codex as run


def main():
    run.verify()
    src, out, root = run.src, run.OUT, run.ROOT
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
    spec = importlib.util.spec_from_file_location("mixed_score_audit_native", root / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                          for name in ("states.pth", "velocities.pth")]
    report = json.loads((out / "report.json").read_text())
    assert report["physical_sha256"] == run.sha(out / "physical.json")
    rows = []
    for reported, seq in zip(report["rows"], (9, 11), strict=True):
        path, source = out / f"root_{seq}", src.OUT / f"root_{seq}"
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            observed, target, history, reference, warm = (f[k].copy() for k in
                ("states", "goal_state", "history_states", "reference", "warm"))
        with np.load(source / "physical.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(observed[:32], f["states"][:32])
            for key, expected in (("goal_state", target), ("history_states", history),
                                  ("reference", reference), ("warm", warm)):
                np.testing.assert_array_equal(f[key], expected)
        with np.load(path / "input/images.npz", allow_pickle=False) as f:
            frames, goal_rgb = f["terminal_rgb"].copy(), f["goal_rgb"].copy()
        with np.load(source / "input/images.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(frames[:32], f["terminal_rgb"][:32])
            np.testing.assert_array_equal(goal_rgb, f["goal_rgb"])
        with np.load(path / "input/context.npz", allow_pickle=False) as f:
            rgb = f["rgb_history"].copy()
            np.testing.assert_array_equal(f["goal_rgb"][0], goal_rgb)
        rng = np.random.default_rng(2026110710 + seq)
        noise = np.concatenate((rng.uniform(-.015, .015, (16, 15, 2)),
                                rng.uniform(-.15, .15, (16, 15, 2))))
        bank = np.concatenate((np.load(source / "input/bank.npy", allow_pickle=False)[:32],
                               np.clip(reference[None] + noise, -1, 1).astype(np.float32)))
        np.testing.assert_array_equal(bank, np.load(path / "input/bank.npy", allow_pickle=False))
        assert len({x.tobytes() for x in bank}) == 64
        assert all(not np.array_equal(x, reference) for x in bank)
        initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
        for j in range(32, 64):
            env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
            try:
                env.seed(0)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], rgb[0])
                for step, action in enumerate(warm):
                    obs, _, done, info = env.step(action.astype(np.float64))
                    assert not done
                    if step in (4, 9):
                        h = 1 if step == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], rgb[h])
                for action in bank[j]:
                    obs, _, done, info = env.step(action.astype(np.float64))
                    assert not done
                np.testing.assert_array_equal(info["state"], observed[j])
                np.testing.assert_array_equal(obs["visual"], frames[j])
            finally:
                env.close()
        receipt = json.loads((path / "output/receipt.json").read_text())
        assert receipt["status"] == "completed" and receipt["worker_sha256"] == run.sha(src.WORKER)
        assert receipt["latents_sha256"] == run.sha(path / "output/latents.npz")
        for key, filename in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                              ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            assert receipt[key] == run.sha(path / "input" / filename)
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        joint = np.linalg.norm(observed[:, :4] - target[:4], axis=1)
        angle = np.abs((observed[:, 4]-target[4]+np.pi) % (2*np.pi)-np.pi)
        success = (joint < 20) & (angle < np.pi/9)
        assert success.tolist() == reported["success_by_index"]
        assert int(np.sum(np.any((observed[:, :4] < 40) | (observed[:, :4] > 472), axis=1))) == reported["candidate_center_outside"]
        scores = {"predicted": ((pred-goal)**2).mean((1, 2)),
                  "actual_encoded": ((actual-goal)**2).mean((1, 2))}
        for name, indices in (("all", np.arange(64)), ("independent", np.arange(32)),
                              ("near", np.arange(32, 48)), ("wide", np.arange(48, 64))):
            group = reported["groups"][name]
            assert group["count"] == len(indices) and group["success_count"] == int(success[indices].sum())
            for label, values in scores.items():
                ranks = indices[np.argsort(values[indices], kind="stable")]
                hits = np.flatnonzero(success[ranks])
                g, b = values[indices][success[indices]], values[indices][~success[indices]]
                row = group[label]
                assert row["index"] == int(ranks[0]) and row["success"] == bool(success[ranks[0]])
                assert row["first_success_rank"] == (int(hits[0])+1 if len(hits) else None)
                assert row["success_failure_pair_win"] == int(np.sum(g[:, None] < b[None, :]))
                assert row["success_failure_pair_tie"] == int(np.sum(g[:, None] == b[None, :]))
                assert row["success_failure_pairs"] == len(g)*len(b)
        rows.append({"sequence": seq, "conditional_replayed": 32, "independent_reused_checked": 32})
        print(f"sequence {seq}: candidates, replay, labels and rankings checked", flush=True)
    run.verify()
    with (out / "audit.json").open("x") as f:
        json.dump({"status": "passed", "auditor_sha256": run.sha(Path(__file__)),
                   "report_sha256": run.sha(out / "report.json"), "rows": rows,
                   "scope": "conditional same-backend replay, source reuse checked, no independent neural rerun"},
                  f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
