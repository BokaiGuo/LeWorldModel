"""Privileged reference-conditioned scoring diagnostic; never a planner input."""
import argparse
import importlib.util
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_continuous_warmup_score_dev_20261105_codex as src

ROOT, sha = src.ROOT, src.sha
OUT = ROOT / "results/pc_pusht_reference_mixed_score_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_REFERENCE_MIXED_SCORE_DEV_PROTOCOL_20261105_CODEX.md"


def make_bank(sequence, reference, independent):
    rng = np.random.default_rng(2026110710 + sequence)
    a = rng.uniform(-.015, .015, (16, 15, 2))
    b = rng.uniform(-.15, .15, (16, 15, 2))
    conditioned = np.clip(reference[None] + np.concatenate((a, b)), -1, 1).astype(np.float32)
    bank = np.concatenate((independent[:32], conditioned))
    if bank.shape != (64, 15, 2) or len({x.tobytes() for x in bank}) != 64 or \
            any(np.array_equal(chain, reference) for chain in bank):
        raise ValueError("invalid or duplicate candidate")
    return bank


def physical():
    if OUT.exists():
        raise FileExistsError(OUT)
    src.verify()
    overlay = (ROOT / ".jepa-physics-site").resolve()
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("exact physics overlay required")
    import gym
    import pymunk
    import skimage
    if ((gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2")
            or any(not Path(m.__file__).resolve().is_relative_to(overlay) for m in (gym, pymunk, skimage))):
        raise ValueError("physics dependency mismatch")
    base = src.visible.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("mixed_score_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    states, velocities = [torch.load(src.DATA / name, map_location="cpu", weights_only=True).numpy()
                          for name in ("states.pth", "velocities.pth")]
    dependencies = [Path(__file__), PROTOCOL, Path(src.__file__), src.OUT / "physical.json",
                    src.WORKER, ROOT / runtime.ENV_REL]
    bindings = {str(p): sha(p) for p in dependencies}
    OUT.mkdir()
    summary = {"status": "running", "bindings": bindings, "rows": []}
    try:
        for seq in (9, 11):
            source, path = src.OUT / f"root_{seq}", OUT / f"root_{seq}"
            (path / "input").mkdir(parents=True)
            (path / "output").mkdir()
            with np.load(source / "physical.npz", allow_pickle=False) as f:
                old_states, target, history, reference, warm = (f[k].copy() for k in
                    ("states", "goal_state", "history_states", "reference", "warm"))
            with np.load(source / "input/images.npz", allow_pickle=False) as f:
                old_images, goal_rgb = f["terminal_rgb"].copy(), f["goal_rgb"].copy()
            with np.load(source / "input/context.npz", allow_pickle=False) as f:
                rgb = f["rgb_history"].copy()
                np.testing.assert_array_equal(f["goal_rgb"][0], goal_rgb)
            independent = np.load(source / "input/bank.npy", allow_pickle=False)
            fixed = make_bank(seq, reference, independent)
            initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
            terminal, final = list(old_images[:32]), list(old_states[:32])
            for chain in fixed[32:]:
                env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
                try:
                    env.seed(0)
                    obs, state = env.reset()
                    np.testing.assert_array_equal(obs["visual"], rgb[0])
                    np.testing.assert_array_equal(state, history[0])
                    for step, action in enumerate(warm):
                        obs, _, done, info = env.step(action.astype(np.float64))
                        if done:
                            raise ValueError("warmup early done")
                        if step in (4, 9):
                            h = 1 if step == 4 else 2
                            np.testing.assert_array_equal(info["state"], history[h])
                            np.testing.assert_array_equal(obs["visual"], rgb[h])
                    for action in chain:
                        obs, _, done, info = env.step(action.astype(np.float64))
                        if done:
                            raise ValueError("candidate early done")
                    terminal.append(obs["visual"].copy())
                    final.append(info["state"].copy())
                finally:
                    env.close()
            for name in ("context.npz", "model_metadata.json"):
                shutil.copyfile(source / "input" / name, path / "input" / name)
            with (path / "input/bank.npy").open("xb") as f:
                np.save(f, fixed)
            with (path / "input/images.npz").open("xb") as f:
                np.savez_compressed(f, goal_rgb=goal_rgb, terminal_rgb=np.stack(terminal))
            with (path / "physical.npz").open("xb") as f:
                np.savez_compressed(f, states=np.stack(final), goal_state=target, history_states=history,
                                    reference=reference, warm=warm)
            summary["rows"].append({"sequence": seq, "physical_sha256": sha(path / "physical.npz"),
                                    "inputs": {p.name: sha(p) for p in sorted((path / "input").iterdir())}})
            print(f"sequence {seq}: 32 reused + 32 conditional trajectories", flush=True)
        src.verify()
        if any(sha(Path(p)) != h for p, h in bindings.items()):
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
    src.verify()
    r = json.loads((OUT / "physical.json").read_text())
    if r["status"] != "completed" or [x["sequence"] for x in r["rows"]] != [9, 11]:
        raise ValueError("incomplete physical phase")
    if any(sha(Path(p)) != h for p, h in r["bindings"].items()):
        raise ValueError("dependency changed")
    for row in r["rows"]:
        path = OUT / f"root_{row['sequence']}"
        if sha(path / "physical.npz") != row["physical_sha256"] or any(
                sha(path / "input" / name) != h for name, h in row["inputs"].items()):
            raise ValueError("artifact changed")


def neural():
    verify()
    for seq in (9, 11):
        src.visible.core.old.neural.launch(OUT / f"root_{seq}", "experiments.pc_jepa_v22_contact_latent_error_worker_20261105_codex")
        print(f"sequence {seq}: neural scoring complete", flush=True)


def summarize():
    verify()
    rows = []
    for seq in (9, 11):
        path = OUT / f"root_{seq}"
        receipt = json.loads((path / "output/receipt.json").read_text())
        if receipt["status"] != "completed" or receipt["worker_sha256"] != sha(src.WORKER) or \
                receipt["latents_sha256"] != sha(path / "output/latents.npz"):
            raise ValueError("neural receipt")
        for key, name in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                          ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            if receipt[key] != sha(path / "input" / name):
                raise ValueError("neural input mismatch")
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target = f["states"].copy(), f["goal_state"].copy()
        joint = np.linalg.norm(states[:, :4] - target[:4], axis=1)
        angle = np.abs((states[:, 4]-target[4]+np.pi) % (2*np.pi)-np.pi)
        success = (joint < 20) & (angle < np.pi/9)
        scores = {"predicted": ((pred-goal)**2).mean(axis=(1, 2)),
                  "actual_encoded": ((actual-goal)**2).mean(axis=(1, 2))}
        groups = {}
        for group, indices in (("all", np.arange(64)), ("independent", np.arange(32)),
                               ("near", np.arange(32, 48)), ("wide", np.arange(48, 64))):
            entries = {}
            for name, values in scores.items():
                order = indices[np.argsort(values[indices], kind="stable")]
                hits = np.flatnonzero(success[order])
                good, bad = values[indices][success[indices]], values[indices][~success[indices]]
                entries[name] = {"index": int(order[0]), "success": bool(success[order[0]]),
                                 "first_success_rank": int(hits[0]+1) if len(hits) else None,
                                 "success_failure_pair_win": int(np.sum(good[:, None] < bad[None, :])),
                                 "success_failure_pair_tie": int(np.sum(good[:, None] == bad[None, :])),
                                 "success_failure_pairs": int(len(good)*len(bad))}
            groups[group] = {"count": len(indices), "success_count": int(success[indices].sum()), **entries}
        rows.append({"sequence": seq, "success_by_index": success.tolist(),
                     "candidate_center_outside": int(np.sum(np.any((states[:, :4] < 40) | (states[:, :4] > 472), axis=1))),
                     "groups": groups})
    with (OUT / "report.json").open("x") as f:
        json.dump({"status": "completed", "physical_sha256": sha(OUT / "physical.json"), "rows": rows,
                   "scope": "reference-conditioned privileged scoring diagnostic, not planning"},
                  f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("physical", "neural", "summarize"))
    phase = parser.parse_args().phase
    print(f"Privileged conditional scoring diagnostic: {phase}", flush=True)
    {"physical": physical, "neural": neural, "summarize": summarize}[phase]()
