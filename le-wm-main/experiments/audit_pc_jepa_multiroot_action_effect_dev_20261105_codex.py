"""Independent arithmetic and same-backend physical replay for multi-root diagnostic."""
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np

from experiments import pc_jepa_multiroot_action_effect_dev_20261105_codex as run

ROOT, OUT, sha = run.ROOT, run.OUT, run.sha


def order(x, y):
    agree = disagree = tied = 0
    for a, b in itertools.combinations(range(len(x)), 2):
        dx, dy = x[a] - x[b], y[a] - y[b]
        if dx == 0 or dy == 0:
            tied += 1
        elif np.sign(dx) == np.sign(dy):
            agree += 1
        else:
            disagree += 1
    return agree, disagree, tied


def main():
    audit_path = OUT / "audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    run.verify()
    report = json.loads((OUT / "report.json").read_text())
    if report["status"] != "completed" or report["physical_sha256"] != sha(OUT / "physical.json"):
        raise ValueError("report binding")
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    import gym
    import pymunk
    import skimage
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))):
        raise ValueError("physics version or origin")
    base = run.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("multiroot_audit_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    expected = np.repeat(np.random.default_rng(2026110590).uniform(-1, 1, (64, 3, 2)).astype(np.float32), 5, axis=1)
    expected[0] = 0
    bank = np.load(OUT / "bank.npy", allow_pickle=False)
    np.testing.assert_array_equal(bank, expected)
    warm = np.asarray(run.old.prior.SCHEDULE, np.float64).reshape(15, 2)[:10]
    rows = []
    for index, row in enumerate(report["rows"]):
        path = OUT / f"root_{index}"
        receipt = json.loads((path / "output/receipt.json").read_text())
        if (receipt["status"] != "completed" or receipt["worker_sha256"] != sha(run.WORKER)
                or receipt["latents_sha256"] != sha(path / "output/latents.npz")
                or row["latents_sha256"] != receipt["latents_sha256"]):
            raise ValueError("worker receipt mismatch")
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, current, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "current", "goal"))
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target, history = (f[k].copy() for k in ("states", "goal_state", "history_states"))
        with np.load(path / "input/images.npz", allow_pickle=False) as f:
            goal_rgb, frames = f["goal_rgb"].copy(), f["terminal_rgb"].copy()
        with np.load(path / "input/context.npz", allow_pickle=False) as f:
            context = {k: f[k].copy() for k in f.files}
        goal_env = native.PushTEnv(**base.CTOR)
        try:
            goal_env.seed(run.GOAL_SEEDS[index])
            obs, state = goal_env.reset()
            np.testing.assert_array_equal(obs["visual"], goal_rgb)
            np.testing.assert_array_equal(state, target)
        finally:
            goal_env.close()
        for j, chain in enumerate(bank):
            env = native.PushTEnv(**base.CTOR)
            try:
                env.seed(run.ROOT_SEEDS[index])
                obs, state = env.reset()
                np.testing.assert_array_equal(obs["visual"], context["rgb_history"][0])
                np.testing.assert_array_equal(state, history[0])
                for step, action in enumerate(warm):
                    obs, _, done, info = env.step(action.copy())
                    if done:
                        raise ValueError("warmup terminated")
                    if step in (4, 9):
                        h = 1 if step == 4 else 2
                        np.testing.assert_array_equal(obs["visual"], context["rgb_history"][h])
                        np.testing.assert_array_equal(info["state"], history[h])
                for action in chain.astype(np.float64):
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("chain terminated")
                np.testing.assert_array_equal(obs["visual"], frames[j])
                np.testing.assert_array_equal(info["state"], states[j])
            finally:
                env.close()
        pred_score = ((pred - goal) ** 2).mean(axis=(1, 2))
        actual_score = ((actual - goal) ** 2).mean(axis=(1, 2))
        pe = ((pred - actual) ** 2).mean(axis=(1, 2))
        persistence = ((current - actual) ** 2).mean(axis=(1, 2))
        obj = np.linalg.norm(states[:, 2:4] - target[2:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2 * np.pi) - np.pi)
        for name, x, y in (("predicted_vs_actual_score", pred_score, actual_score),
                           ("predicted_vs_object", pred_score, obj), ("actual_vs_object", actual_score, obj),
                           ("predicted_vs_angle", pred_score, angle), ("actual_vs_angle", actual_score, angle)):
            observed = row[name]
            if order(x, y) != (observed["agree"], observed["disagree"], observed["tied"]):
                raise ValueError(f"order mismatch {index} {name}")
        if (abs(float(pe.mean()) - row["mean_prediction_mse"]) > 1e-12
                or abs(float(persistence.mean()) - row["mean_persistence_mse"]) > 1e-12
                or int((pe < persistence).sum()) != row["prediction_beats_persistence"]):
            raise ValueError("MSE mismatch")
        rows.append({"root": index, "replayed": len(bank),
                     "different_from_zero_object_position_1e-6": int((np.linalg.norm(states[:, 2:4] - states[0, 2:4], axis=1) > 1e-6).sum()),
                     "zero_tail_object_position_drift": float(np.linalg.norm(states[0, 2:4] - history[-1, 2:4]))})
        print(f"root {index}: 64 physical replays and arithmetic verified", flush=True)
    run.verify()
    result = {"status": "passed", "report_sha256": sha(OUT / "report.json"),
              "physical_sha256": sha(OUT / "physical.json"), "auditor_sha256": sha(Path(__file__)),
              "rows": rows, "scope": "same backend; no independent neural recomputation"}
    with audit_path.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
