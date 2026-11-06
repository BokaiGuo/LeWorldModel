"""Four-root development diagnostic: fixed unprivileged actions, no planning."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from experiments import pc_jepa_v22_unqueried_suffix_dev_20261105_codex as old
from experiments import pc_jepa_v12_contact_score_bank_neural_posthoc_20261104_codex as neural
from experiments.audit_pc_jepa_v22_unqueried_suffix_dev_20261105_codex import independent_order

ROOT, sha = old.ROOT, old.sha
OUT = ROOT / "results/pc_jepa_multiroot_action_effect_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_MULTIR_ROOT_ACTION_EFFECT_DEV_PROTOCOL_20261105_CODEX.md"
WORKER = ROOT / "experiments/pc_jepa_v22_contact_latent_error_worker_20261105_codex.py"
META = old.SOURCE / "response_0/input/model_metadata.json"
ROOT_SEEDS = (2026110511, 2026110512, 2026110513, 2026110514)
GOAL_SEEDS = (2026110515, 2026110516, 2026110517, 2026110518)


def action_bank():
    rng = np.random.default_rng(2026110590)
    bank = np.repeat(rng.uniform(-1, 1, (64, 3, 2)).astype(np.float32), 5, axis=1)
    bank[0] = 0
    if len({x.tobytes() for x in bank}) != 64:
        raise ValueError("duplicate actions")
    return bank


def physical():
    if OUT.exists():
        raise FileExistsError(OUT)
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("physics overlay required")
    import gym
    import pymunk
    import skimage
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))):
        raise ValueError("physics version or origin")
    base = old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("multiroot_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    stats = old.prior.native_stats(ROOT)
    warm = np.asarray(old.prior.SCHEDULE, np.float64).reshape(15, 2)[:10]
    bank = action_bank()
    dependencies = [PROTOCOL, Path(__file__), WORKER, META, ROOT / runtime.ENV_REL,
                    ROOT / "external/jepa-wms/app/plan_common/datasets/pusht_dset.py"]
    bindings = {str(p): sha(p) for p in dependencies}
    OUT.mkdir()
    summary = {"status": "running", "bindings": bindings, "roots": [],
               "root_seeds": ROOT_SEEDS, "goal_seeds": GOAL_SEEDS}
    try:
        with (OUT / "bank.npy").open("xb") as handle:
            np.save(handle, bank)
        for number, (root_seed, goal_seed) in enumerate(zip(ROOT_SEEDS, GOAL_SEEDS, strict=True)):
            directory = OUT / f"root_{number}"
            (directory / "input").mkdir(parents=True)
            (directory / "output").mkdir()
            goal_env = native.PushTEnv(**base.CTOR)
            try:
                goal_env.seed(goal_seed)
                goal_obs, goal_state = goal_env.reset()
                goal_rgb, goal_state = goal_obs["visual"].copy(), goal_state.copy()
            finally:
                goal_env.close()
            histories, terminal_rgb, states = None, [], []
            for index, chain in enumerate(bank):
                env = native.PushTEnv(**base.CTOR)
                try:
                    env.seed(root_seed)
                    obs, state = env.reset()
                    images, props, start_states = [obs["visual"].copy()], [obs["proprio"].copy()], [state.copy()]
                    for step, action in enumerate(warm):
                        obs, _, done, info = env.step(action.copy())
                        if done:
                            raise ValueError(f"early termination in warmup root={number} action={index} step={step}")
                        if step in (4, 9):
                            images.append(obs["visual"].copy())
                            props.append(obs["proprio"].copy())
                            start_states.append(info["state"].copy())
                    current = (np.stack(images), np.stack(props), np.stack(start_states))
                    if histories is None:
                        histories = current
                    else:
                        for actual, expected in zip(current, histories, strict=True):
                            np.testing.assert_array_equal(actual, expected)
                    for step, action in enumerate(chain.astype(np.float64)):
                        obs, _, done, info = env.step(action)
                        if done:
                            raise ValueError(f"early termination root={number} action={index} step={step}")
                    terminal_rgb.append(obs["visual"].copy())
                    states.append(info["state"].copy())
                finally:
                    env.close()
            rgb, props, start_states = histories
            prop = (props.astype(np.float32) - stats["PROPRIO_MEAN"]) / stats["PROPRIO_STD"]
            history_u = (100 * warm).astype(np.float32) / np.float32(100)
            packets = ((history_u - stats["ACTION_MEAN"]) / stats["ACTION_STD"]).reshape(2, 10)
            with (directory / "input/context.npz").open("xb") as handle:
                np.savez_compressed(handle, rgb_history=rgb, goal_rgb=goal_rgb[None],
                                    normalized_proprio=prop, normalized_history_packets=packets,
                                    action_mean=stats["ACTION_MEAN"], action_std=stats["ACTION_STD"],
                                    carry=np.empty((0, 2), np.float32))
            shutil.copyfile(OUT / "bank.npy", directory / "input/bank.npy")
            shutil.copyfile(META, directory / "input/model_metadata.json")
            with (directory / "input/images.npz").open("xb") as handle:
                np.savez_compressed(handle, goal_rgb=goal_rgb, terminal_rgb=np.stack(terminal_rgb))
            with (directory / "physical.npz").open("xb") as handle:
                np.savez_compressed(handle, states=np.stack(states), goal_state=goal_state,
                                    history_states=start_states)
            summary["roots"].append({"root": number, "status": "completed", "count": len(states),
                                     "physical_sha256": sha(directory / "physical.npz"),
                                     "input_sha256": {p.name: sha(p) for p in sorted((directory / "input").iterdir())}})
            print(f"root {number}: 64 legal action chains replayed", flush=True)
        if any(sha(Path(p)) != value for p, value in bindings.items()):
            raise ValueError("dependency changed")
        summary.update(status="completed", bank_sha256=sha(OUT / "bank.npy"))
    except Exception:
        summary.update(status="failed", error=traceback.format_exc())
    with (OUT / "physical.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if summary["status"] != "completed":
        raise RuntimeError(summary["error"])


def verify():
    receipt = json.loads((OUT / "physical.json").read_text())
    if (receipt["status"] != "completed" or sha(OUT / "bank.npy") != receipt["bank_sha256"]
            or any(sha(Path(p)) != v for p, v in receipt["bindings"].items())):
        raise ValueError("source binding mismatch")
    for row in receipt["roots"]:
        path = OUT / f"root_{row['root']}"
        if (sha(path / "physical.npz") != row["physical_sha256"]
                or any(sha(path / "input" / p) != digest for p, digest in row["input_sha256"].items())):
            raise ValueError("root binding mismatch")
    return receipt


def neural_score():
    verify()
    for i in range(len(ROOT_SEEDS)):
        path = OUT / f"root_{i}"
        if (path / "output/receipt.json").exists():
            raise FileExistsError(path / "output/receipt.json")
        neural.launch(path, "experiments.pc_jepa_v22_contact_latent_error_worker_20261105_codex")
        print(f"root {i}: predicted and encoded", flush=True)


def summarize():
    verify()
    rows = []
    for i in range(len(ROOT_SEEDS)):
        path = OUT / f"root_{i}"
        receipt = json.loads((path / "output/receipt.json").read_text())
        if (receipt["status"] != "completed" or receipt["worker_sha256"] != sha(WORKER)
                or receipt["latents_sha256"] != sha(path / "output/latents.npz")
                or any(receipt[k] != sha(path / "input" / name) for k, name in (
                    ("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                    ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")))):
            raise ValueError("worker output binding mismatch")
        with np.load(path / "output/latents.npz", allow_pickle=False) as data:
            pred, actual, current, goal = (data[k].astype(np.float64) for k in ("predicted", "actual", "current", "goal"))
        with np.load(path / "physical.npz", allow_pickle=False) as data:
            states, target, start = (data[k] for k in ("states", "goal_state", "history_states"))
        pred_score = ((pred - goal) ** 2).mean(axis=(1, 2))
        actual_score = ((actual - goal) ** 2).mean(axis=(1, 2))
        pred_error = ((pred - actual) ** 2).mean(axis=(1, 2))
        persistence = ((current - actual) ** 2).mean(axis=(1, 2))
        position = np.linalg.norm(states[:, 2:4] - target[2:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2 * np.pi) - np.pi)
        rows.append({"root": i, "count": len(states),
                     "object_pose_count_exact": len(np.unique(states[:, 2:5], axis=0)),
                     "object_moved_count_1e-6": int(np.sum(np.linalg.norm(states[:, 2:4] - start[-1, 2:4], axis=1) > 1e-6)),
                     "mean_prediction_mse": float(pred_error.mean()), "mean_persistence_mse": float(persistence.mean()),
                     "prediction_beats_persistence": int((pred_error < persistence).sum()),
                     "predicted_vs_actual_score": independent_order(pred_score, actual_score),
                     "predicted_vs_object": independent_order(pred_score, position),
                     "actual_vs_object": independent_order(actual_score, position),
                     "predicted_vs_angle": independent_order(pred_score, angle),
                     "actual_vs_angle": independent_order(actual_score, angle),
                     "latents_sha256": sha(path / "output/latents.npz")})
    report = {"status": "completed", "physical_sha256": sha(OUT / "physical.json"),
              "rows": rows, "scope": "development only; source independence unknown; queried actions"}
    with (OUT / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("physical", "neural", "summarize"))
    phase = parser.parse_args().phase
    print(f"Four-root nonprivileged action diagnostic: {phase}", flush=True)
    {"physical": physical, "neural": neural_score, "summarize": summarize}[phase]()


if __name__ == "__main__":
    main()
