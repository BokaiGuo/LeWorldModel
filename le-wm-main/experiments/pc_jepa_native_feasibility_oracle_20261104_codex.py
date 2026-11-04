"""Fixed-budget privileged physics-only feasibility diagnosis for an exposed root."""
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from experiments import pc_jepa_native_feedback_dev_20261104_codex as prior

ROOT, SOURCE = prior.ROOT, prior.OUT
OUT = ROOT / "results/pc_jepa_native_feasibility_oracle_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_NATIVE_FEASIBILITY_ORACLE_PROTOCOL_20261104_CODEX.md"


def main():
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("physics overlay required")
    import gym
    import pymunk
    import skimage
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("physics version mismatch")
    if any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage)):
        raise ValueError("physics module origin mismatch")
    base = prior.prior.helpers(ROOT)
    runtime, _ = base.load_runtime(ROOT)
    spec = importlib.util.spec_from_file_location("physical_oracle_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    replay = json.loads((SOURCE / "physical_replay.json").read_text())
    if replay["status"] != "exact" or replay["summary_sha256"] != prior.sha(SOURCE / "summary.json"):
        raise ValueError("unverified source replay")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    target = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False)
    paths = [PROTOCOL, Path(__file__), ROOT / runtime.ENV_REL, SOURCE / "summary.json",
             SOURCE / "manifest.json", SOURCE / "physical_replay.json", SOURCE / "goal_state_evaluation_only.npy"]
    anchors = {}
    for arm in ("response", "cem"):
        commands = [SOURCE / f"{arm}_physical/command_{i:02d}.json" for i in range(11, 26)]
        paths.extend(commands)
        anchors[arm] = np.asarray([json.loads(p.read_text())["action"] for p in commands], np.float64)
        if anchors[arm].shape != (15, 2) or np.any(np.abs(anchors[arm]) > 1):
            raise ValueError("invalid saved actions")
    warm = np.asarray(manifest["warmup"], np.float64)
    frame10 = SOURCE / "response_physical/frame_10"
    paths.extend(frame10 / f"{k}.npy" for k in ("rgb", "state", "proprio"))
    expected = {k: np.load(frame10 / f"{k}.npy", allow_pickle=False) for k in ("rgb", "state", "proprio")}
    bindings = {str(p): prior.sha(p) for p in paths}
    OUT.mkdir()
    result = {"status": "running", "protocol_sha256": prior.sha(PROTOCOL), "bindings": bindings,
              "scope": "single exposed root; privileged open-loop feasibility diagnosis only",
              "rows": [], "generations": [], "random_seed": 20261104}
    started, error = time.monotonic(), None
    best, best_data, first_success, success_data, actions_all = None, None, None, None, []
    try:
        rng = np.random.default_rng(20261104)
        with (OUT / "events.jsonl").open("x", buffering=1) as events:
            def evaluate(actions, label):
                nonlocal best, best_data, first_success, success_data
                if actions.shape != (15, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                    raise ValueError("invalid actions")
                env = native.PushTEnv(**base.CTOR)
                env.seed(prior.prior.SEED)
                obs, state = env.reset()
                for action in warm:
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("early termination in warmup")
                    state = info["state"]
                for key, observed in (("rgb", obs["visual"]), ("state", state), ("proprio", obs["proprio"])):
                    np.testing.assert_array_equal(observed, expected[key])
                frames, states, props = [obs["visual"].copy()], [state.copy()], [obs["proprio"].copy()]
                for action in actions:
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("unexpected termination")
                    state = info["state"]
                    frames.append(obs["visual"].copy())
                    states.append(state.copy())
                    props.append(obs["proprio"].copy())
                env.close()
                m = prior.metrics(state, target)
                cost = (m["joint_xy_error"] / 20.) ** 2 + (m["angle_error_radians"] / (np.pi / 9.)) ** 2
                index = len(actions_all)
                row = {"index": index, "label": label, "cost": float(cost), "metrics": m,
                       "terminal_state": state.tolist()}
                actions_all.append(actions.copy())
                result["rows"].append(row)
                events.write(json.dumps(row, allow_nan=False) + "\n")
                if best is None or cost < best["cost"]:
                    best, best_data = row, {"actions": actions.copy(), "rgb": np.stack(frames),
                                            "state": np.stack(states), "proprio": np.stack(props)}
                if m["success"] and first_success is None:
                    first_success, success_data = row, {"actions": actions.copy(), "rgb": np.stack(frames),
                                                        "state": np.stack(states), "proprio": np.stack(props)}
                return cost

            for arm in ("response", "cem"):
                evaluate(anchors[arm], f"anchor_{arm}")
            for arm in ("zero", "response", "cem"):
                mean = np.zeros((15, 2)) if arm == "zero" else anchors[arm].copy()
                sigma = np.full((15, 2), 0.5)
                for generation in range(4):
                    samples = np.clip(rng.normal(mean, sigma, size=(16, 15, 2)), -1, 1)
                    costs = np.asarray([evaluate(a, f"{arm}_{generation}_{j}") for j, a in enumerate(samples)])
                    elite = samples[np.argsort(costs, kind="stable")[:4]]
                    result["generations"].append({"arm": arm, "generation": generation,
                        "mean_before": mean.tolist(), "sigma_before": sigma.tolist(),
                        "best_cost": float(costs.min()), "elite_indices": np.argsort(costs, kind="stable")[:4].tolist()})
                    mean, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.05)
                    print(f"{arm} gen {generation}: best={costs.min():.3f}, global={best['cost']:.3f}", flush=True)
        if len(actions_all) != 194 or any(prior.sha(Path(p)) != h for p, h in bindings.items()):
            raise ValueError("candidate count or binding mismatch")
        with (OUT / "actions.npy").open("xb") as f:
            np.save(f, np.stack(actions_all), allow_pickle=False)
        with (OUT / "best_trajectory.npz").open("xb") as f:
            np.savez_compressed(f, **best_data)
        if first_success is not None:
            with (OUT / "success_trajectory.npz").open("xb") as f:
                np.savez_compressed(f, **success_data)
        result.update(status="complete", best_index=best["index"], successful_indices=[
            row["index"] for row in result["rows"] if row["metrics"]["success"]],
            actions_sha256=prior.sha(OUT / "actions.npy"),
            best_trajectory_sha256=prior.sha(OUT / "best_trajectory.npz"),
            success_trajectory_sha256=(prior.sha(OUT / "success_trajectory.npz") if first_success else None))
    except Exception:
        error = traceback.format_exc()
        result.update(status="failed", error=error)
    result["wall_seconds"] = time.monotonic() - started
    with (OUT / "result.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
