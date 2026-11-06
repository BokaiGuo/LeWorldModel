"""Same-backend physical replay and independent arithmetic for reachable targets."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_jepa_reachable_random_goal_dev_20261105_codex as run
from experiments.audit_pc_jepa_multiroot_action_effect_dev_20261105_codex import order


def main():
    out, root, sha = run.OUT, run.ROOT, run.sha
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    run.verify()
    report = json.loads((out / "report.json").read_text())
    assert report["physical_sha256"] == sha(out / "physical.json")
    overlay = (root / ".jepa-physics-site").resolve()
    assert os.environ.get("PYTHONPATH") == str(overlay)
    import gym
    import pymunk
    import skimage
    assert (gym.__version__, pymunk.version, skimage.__version__) == ("0.23.1", "6.8.0", "0.25.2")
    assert all(Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))
    base = run.old.old.prior.helpers(root)
    runtime, parent = base.load_runtime(root)
    assert runtime.probe(root, parent, root / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] == \
        "native_import_ready_physics_not_executed"
    spec = importlib.util.spec_from_file_location("reachable_audit_native", root / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    generated = np.repeat(np.random.default_rng(2026110610).uniform(-1, 1, (65, 3, 2))
                          .astype(np.float32), 5, axis=1)
    np.testing.assert_array_equal(generated[64], np.load(out / "evaluation_reference.npy"))
    warm = np.asarray(run.old.old.prior.SCHEDULE, np.float64).reshape(15, 2)[:10]
    rows = []
    for number, seed in enumerate(run.SEEDS):
        path = out / f"root_{number}"
        np.testing.assert_array_equal(np.load(path / "input/bank.npy"), generated[:64])
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target, history = (f[k].copy() for k in ("states", "goal_state", "history_states"))
        with np.load(path / "input/images.npz", allow_pickle=False) as f:
            frames, goal_rgb = f["terminal_rgb"].copy(), f["goal_rgb"].copy()
        with np.load(path / "input/context.npz", allow_pickle=False) as f:
            context = {k: f[k].copy() for k in f.files}
        np.testing.assert_array_equal(context["goal_rgb"][0], goal_rgb)
        for j, chain in enumerate(generated):
            env = native.PushTEnv(**base.CTOR)
            try:
                env.seed(seed)
                obs, state = env.reset()
                np.testing.assert_array_equal(state, history[0])
                np.testing.assert_array_equal(obs["visual"], context["rgb_history"][0])
                for step, action in enumerate(warm):
                    obs, _, done, info = env.step(action.copy())
                    assert not done
                    if step in (4, 9):
                        h = 1 if step == 4 else 2
                        np.testing.assert_array_equal(info["state"], history[h])
                        np.testing.assert_array_equal(obs["visual"], context["rgb_history"][h])
                for action in chain.astype(np.float64):
                    obs, _, done, info = env.step(action)
                    assert not done
                np.testing.assert_array_equal(info["state"], target if j == 64 else states[j])
                np.testing.assert_array_equal(obs["visual"], goal_rgb if j == 64 else frames[j])
            finally:
                env.close()
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        p = ((pred-goal)**2).mean(axis=(1, 2))
        a = ((actual-goal)**2).mean(axis=(1, 2))
        dist = np.linalg.norm(states[:, :4] - target[:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2*np.pi) - np.pi)
        ok = (dist < 20) & (angle < np.pi/9)
        r = report["rows"][number]
        assert ok.tolist() == r["success_by_index"] and int(ok.sum()) == r["success_count"]
        assert order(p, a) == tuple(r["predicted_vs_actual_score"][k] for k in ("agree", "disagree", "tied"))
        for name, scores in (("predicted", p), ("actual_encoded", a)):
            idx = int(np.argmin(scores))
            assert idx == r["choices"][name]["index"] and bool(ok[idx]) == r["choices"][name]["success"]
        # Canvas is 512 square; this is a coordinate diagnostic, not a safety assertion.
        rows.append({"root": number, "replayed": 65, "target_agent": target[:2].tolist(),
                     "target_agent_outside_512": bool(np.any((target[:2] < 0) | (target[:2] > 512))),
                     "candidate_agents_outside_512": int(np.sum(np.any((states[:, :2] < 0) | (states[:, :2] > 512), axis=1)))})
        print(f"root {number}: replayed candidates and target", flush=True)
    run.verify()
    with (out / "audit.json").open("x") as f:
        json.dump({"status": "passed", "report_sha256": sha(out / "report.json"),
                   "auditor_sha256": sha(Path(__file__)), "rows": rows,
                   "scope": "same backend; no independent neural rerun"}, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
