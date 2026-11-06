"""Replay fixed-bank physical results and independently audit saved scoring."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_continuous_warmup_score_dev_20261105_codex as run
from experiments.audit_pc_jepa_multiroot_action_effect_dev_20261105_codex import order


def main():
    run.verify()
    out, root = run.OUT, run.ROOT
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    overlay = (root / ".jepa-physics-site").resolve()
    assert os.environ.get("PYTHONPATH") == str(overlay)
    import gym
    import pymunk
    import skimage
    assert (gym.__version__, pymunk.version, skimage.__version__) == ("0.23.1", "6.8.0", "0.25.2")
    assert all(Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))
    base = run.visible.core.old.old.prior.helpers(root)
    runtime, parent = base.load_runtime(root)
    assert runtime.probe(root, parent, root / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] == \
        "native_import_ready_physics_not_executed"
    spec = importlib.util.spec_from_file_location("continuous_score_audit_native", root / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities, actions = [torch.load(run.DATA / name, map_location="cpu", weights_only=True).numpy()
                                   for name in ("states.pth", "velocities.pth", "rel_actions.pth")]
    packets = np.random.default_rng(2026110702).uniform(-.2, .2, (64, 3, 2)).astype(np.float32)
    packets[0] = 0
    bank = np.repeat(packets, 5, axis=1)
    report = json.loads((out / "report.json").read_text())
    assert report["physical_sha256"] == run.sha(out / "physical.json")
    rows = []
    for report_row, seq in zip(report["rows"], (9, 11), strict=True):
        path = out / f"root_{seq}"
        np.testing.assert_array_equal(bank, np.load(path / "input/bank.npy", allow_pickle=False))
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            observed, target, history = (f[k].copy() for k in ("states", "goal_state", "history_states"))
            np.testing.assert_array_equal(f["warm"], (actions[seq, :10] / 100).astype(np.float32))
            np.testing.assert_array_equal(f["reference"], (actions[seq, 10:25] / 100).astype(np.float32))
        with np.load(path / "input/context.npz", allow_pickle=False) as f:
            images = f["rgb_history"].copy()
            np.testing.assert_array_equal(f["goal_rgb"][0], np.load(path / "input/images.npz")["goal_rgb"])
        with np.load(path / "input/images.npz", allow_pickle=False) as f:
            frames, goal_rgb = f["terminal_rgb"].copy(), f["goal_rgb"].copy()
        initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
        warm = (actions[seq, :10] / 100).astype(np.float32)
        reference = (actions[seq, 10:25] / 100).astype(np.float32)
        for j, chain in enumerate(np.concatenate((bank, reference[None]))):
            env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
            try:
                env.seed(0)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], images[0])
                for step, action in enumerate(warm):
                    obs, _, done, info = env.step(action.astype(np.float64))
                    assert not done
                    if step in (4, 9):
                        h = 1 if step == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], images[h])
                for action in chain:
                    obs, _, done, info = env.step(action.astype(np.float64))
                    assert not done
                np.testing.assert_array_equal(info["state"], target if j == 64 else observed[j])
                np.testing.assert_array_equal(obs["visual"], goal_rgb if j == 64 else frames[j])
            finally:
                env.close()
        receipt = json.loads((path / "output/receipt.json").read_text())
        assert receipt["status"] == "completed"
        assert receipt["worker_sha256"] == run.sha(run.WORKER)
        assert receipt["latents_sha256"] == run.sha(path / "output/latents.npz")
        for key, filename in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                              ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            assert receipt[key] == run.sha(path / "input" / filename)
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        pos = np.linalg.norm(observed[:, :4] - target[:4], axis=1)
        obj = np.linalg.norm(observed[:, 2:4] - target[2:4], axis=1)
        angle = np.abs((observed[:, 4]-target[4]+np.pi) % (2*np.pi)-np.pi)
        ok = (pos < 20) & (angle < np.pi/9)
        p, a = ((pred-goal)**2).mean((1, 2)), ((actual-goal)**2).mean((1, 2))
        assert ok.tolist() == report_row["success_by_index"] and int(ok.sum()) == report_row["success_count"]
        assert order(p, a) == tuple(report_row["predicted_vs_actual"][k] for k in ("agree", "disagree", "tied"))
        for name, values in (("predicted", p), ("actual_encoded", a)):
            ranks = np.argsort(values, kind="stable")
            hits = np.flatnonzero(ok[ranks])
            choice = report_row["choices"][name]
            assert int(ranks[0]) == choice["index"] and bool(ok[ranks[0]]) == choice["success"]
            assert choice["first_success_rank"] == (int(hits[0])+1 if len(hits) else None)
            for key, metric in (("vs_joint_position", pos), ("vs_object_position", obj)):
                assert order(values, metric) == tuple(choice[key][k] for k in ("agree", "disagree", "tied"))
        err = np.max(np.abs(target[:4] - states[seq, 25, :4]))
        rows.append({"sequence": seq, "replayed": 65, "recorded_goal_position_max_abs_difference": float(err)})
        print(f"sequence {seq}: 65 chains and scoring arithmetic checked", flush=True)
    run.verify()
    with (out / "audit.json").open("x") as f:
        json.dump({"status": "passed", "auditor_sha256": run.sha(Path(__file__)),
                   "report_sha256": run.sha(out / "report.json"), "rows": rows,
                   "scope": "same-backend physical replay; no independent neural rerun"}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
