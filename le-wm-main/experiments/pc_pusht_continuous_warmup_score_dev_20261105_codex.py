"""Fixed unconditioned bank scored on continuous dataset-warmup contact goals."""
import argparse
import importlib.util
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import audit_pc_pusht_val_action_replay_dev_20261105_codex as prior
from experiments import pc_jepa_visible_local_bank_dev_20261105_codex as visible

ROOT, sha = prior.ROOT, prior.sha
DATA = prior.DATA
OUT = ROOT / "results/pc_pusht_continuous_warmup_score_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_CONTINUOUS_WARMUP_SCORE_DEV_PROTOCOL_20261105_CODEX.md"
META = visible.core.old.META
WORKER = visible.core.old.WORKER


def bank():
    rng = np.random.default_rng(2026110702)
    packets = rng.uniform(-.2, .2, (64, 3, 2)).astype(np.float32)
    packets[0] = 0
    return np.repeat(packets, 5, axis=1)


def physical():
    if OUT.exists():
        raise FileExistsError(OUT)
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact physics overlay required")
    import gym
    import pymunk
    import skimage
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))):
        raise ValueError("physics dependency mismatch")
    base = visible.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("continuous_score_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    stats = visible.core.old.old.prior.native_stats(ROOT)
    states, velocities, rel = [torch.load(DATA / name, map_location="cpu", weights_only=True).numpy()
                               for name in ("states.pth", "velocities.pth", "rel_actions.pth")]
    dependencies = [Path(__file__), PROTOCOL, WORKER, META, ROOT / runtime.ENV_REL,
                    ROOT / "external/jepa-wms/app/plan_common/datasets/pusht_dset.py",
                    *(DATA / name for name in ("states.pth", "velocities.pth", "rel_actions.pth"))]
    bindings = {str(p): sha(p) for p in dependencies}
    fixed = bank()
    if fixed.shape != (64, 15, 2) or len({a.tobytes() for a in fixed}) != 64:
        raise ValueError("invalid fixed bank")
    OUT.mkdir()
    summary = {"status": "running", "bindings": bindings, "rows": []}
    try:
        for seq in (9, 11):
            path = OUT / f"root_{seq}"
            (path / "input").mkdir(parents=True)
            (path / "output").mkdir()
            initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
            warm = (rel[seq, :10] / 100).astype(np.float32)
            reference = (rel[seq, 10:25] / 100).astype(np.float32)
            histories, terminals, final_states = None, [], []
            for chain in np.concatenate((fixed, reference[None]), axis=0):
                env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
                try:
                    env.seed(0)
                    obs, state = env.reset()
                    imgs, props, histstates = [obs["visual"].copy()], [obs["proprio"].copy()], [state.copy()]
                    for step, action in enumerate(warm):
                        obs, _, done, info = env.step(action.astype(np.float64))
                        if done:
                            raise ValueError("warmup early done")
                        if step in (4, 9):
                            imgs.append(obs["visual"].copy())
                            props.append(obs["proprio"].copy())
                            histstates.append(info["state"].copy())
                    current = (np.stack(imgs), np.stack(props), np.stack(histstates))
                    if histories is None:
                        histories = current
                    else:
                        for left, right in zip(current, histories, strict=True):
                            np.testing.assert_array_equal(left, right)
                    for action in chain:
                        obs, _, done, info = env.step(action.astype(np.float64))
                        if done:
                            raise ValueError("candidate early done")
                    terminals.append(obs["visual"].copy())
                    final_states.append(info["state"].copy())
                finally:
                    env.close()
            rgb, props, history_states = histories
            goal_rgb, goal_state = terminals[-1], final_states[-1]
            history_u = (100 * warm.astype(np.float64)).astype(np.float32) / np.float32(100)
            prop = (props.astype(np.float32) - stats["PROPRIO_MEAN"]) / stats["PROPRIO_STD"]
            packets = ((history_u - stats["ACTION_MEAN"]) / stats["ACTION_STD"]).reshape(2, 10)
            with (path / "input/context.npz").open("xb") as f:
                np.savez_compressed(f, rgb_history=rgb, goal_rgb=goal_rgb[None], normalized_proprio=prop,
                                    normalized_history_packets=packets, action_mean=stats["ACTION_MEAN"],
                                    action_std=stats["ACTION_STD"], carry=np.empty((0, 2), np.float32))
            with (path / "input/bank.npy").open("xb") as f:
                np.save(f, fixed)
            with (path / "input/images.npz").open("xb") as f:
                np.savez_compressed(f, goal_rgb=goal_rgb, terminal_rgb=np.stack(terminals[:64]))
            shutil.copyfile(META, path / "input/model_metadata.json")
            with (path / "physical.npz").open("xb") as f:
                np.savez_compressed(f, states=np.stack(final_states[:64]), goal_state=goal_state,
                                    history_states=history_states, reference=reference, warm=warm)
            summary["rows"].append({"sequence": seq, "physical_sha256": sha(path / "physical.npz"),
                                    "inputs": {p.name: sha(p) for p in sorted((path / "input").iterdir())}})
            print(f"sequence {seq}: physical bank and reference complete", flush=True)
        if any(sha(Path(p)) != digest for p, digest in bindings.items()):
            raise ValueError("dependency changed")
        summary["status"] = "completed"
    except Exception:
        summary.update(status="failed", error=traceback.format_exc())
    with (OUT / "physical.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    if summary["status"] != "completed":
        raise RuntimeError(summary["error"])


def verify():
    record = json.loads((OUT / "physical.json").read_text())
    if record["status"] != "completed" or [r["sequence"] for r in record["rows"]] != [9, 11]:
        raise ValueError("physical phase incomplete")
    if any(sha(Path(p)) != h for p, h in record["bindings"].items()):
        raise ValueError("dependency changed")
    for row in record["rows"]:
        path = OUT / f"root_{row['sequence']}"
        if sha(path / "physical.npz") != row["physical_sha256"] or any(
                sha(path / "input" / name) != h for name, h in row["inputs"].items()):
            raise ValueError("physical artifact changed")
    return record


def neural():
    verify()
    for seq in (9, 11):
        path = OUT / f"root_{seq}"
        visible.core.old.neural.launch(path, "experiments.pc_jepa_v22_contact_latent_error_worker_20261105_codex")
        print(f"sequence {seq}: neural scoring complete", flush=True)


def summarize():
    verify()
    rows = []
    for seq in (9, 11):
        path = OUT / f"root_{seq}"
        receipt = json.loads((path / "output/receipt.json").read_text())
        if receipt["status"] != "completed" or receipt["worker_sha256"] != sha(WORKER) or \
                receipt["latents_sha256"] != sha(path / "output/latents.npz"):
            raise ValueError("neural receipt")
        for key, name in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                          ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            if receipt[key] != sha(path / "input" / name):
                raise ValueError("neural input mismatch")
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target, history = (f[k].copy() for k in ("states", "goal_state", "history_states"))
        pos = np.linalg.norm(states[:, :4] - target[:4], axis=1)
        obj = np.linalg.norm(states[:, 2:4] - target[2:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2*np.pi) - np.pi)
        success = (pos < 20) & (angle < np.pi/9)
        scores = {"predicted": ((pred-goal)**2).mean(axis=(1, 2)),
                  "actual_encoded": ((actual-goal)**2).mean(axis=(1, 2))}
        choices, order = {}, visible.core.old.independent_order
        for name, values in scores.items():
            ranked = np.argsort(values, kind="stable")
            hits = np.flatnonzero(success[ranked])
            choices[name] = {"index": int(ranked[0]), "success": bool(success[ranked[0]]),
                             "first_success_rank": int(hits[0]+1) if len(hits) else None,
                             "vs_joint_position": order(values, pos), "vs_object_position": order(values, obj)}
        rows.append({"sequence": seq, "goal_state": target.tolist(),
                     "goal_object_displacement": float(np.linalg.norm(target[2:4] - history[-1, 2:4])),
                     "goal_centers_qualified": bool(np.all((target[:4] >= 40) & (target[:4] <= 472))),
                     "candidate_center_outside": int(np.sum(np.any((states[:, :4] < 40) | (states[:, :4] > 472), axis=1))),
                     "success_count": int(success.sum()), "success_by_index": success.tolist(),
                     "predicted_vs_actual": order(scores["predicted"], scores["actual_encoded"]), "choices": choices})
    with (OUT / "report.json").open("x") as f:
        json.dump({"status": "completed", "physical_sha256": sha(OUT / "physical.json"), "rows": rows,
                   "scope": "two selected development sequences; not planning or independent confirmation"},
                  f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("physical", "neural", "summarize"))
    phase = parser.parse_args().phase
    print(f"Continuous warmup fixed-bank diagnostic: {phase}", flush=True)
    {"physical": physical, "neural": neural, "summarize": summarize}[phase]()
