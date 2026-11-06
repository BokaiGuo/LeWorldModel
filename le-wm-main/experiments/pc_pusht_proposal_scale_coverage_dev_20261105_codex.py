"""Physical coverage of fixed, reference-independent PushT proposals."""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_continuous_warmup_score_dev_20261105_codex as old

ROOT, DATA = old.ROOT, old.DATA
OUT = ROOT / "results/pc_pusht_proposal_scale_coverage_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_PROPOSAL_SCALE_COVERAGE_DEV_PROTOCOL_20261105_CODEX.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def banks():
    rng = np.random.default_rng(2026110519)
    wide = rng.uniform(-1, 1, (64, 3, 2))
    innovations = rng.normal(0, .2, (64, 3, 2))
    correlated = np.empty_like(innovations)
    correlated[:, 0] = np.clip(innovations[:, 0], -1, 1)
    for t in (1, 2):
        correlated[:, t] = np.clip(.7 * correlated[:, t-1] + .3 * innovations[:, t], -1, 1)
    mixed = np.concatenate((correlated[:32], wide[32:]), axis=0)
    return {name: np.repeat(x.astype(np.float32), 5, axis=1)
            for name, x in (("wide", wide), ("correlated", correlated), ("mixed", mixed))}


def metrics(states, target, history, crossed):
    pos = np.linalg.norm(states[:, :4] - target[:4], axis=1)
    angle = np.abs((states[:, 4] - target[4] + np.pi) % (2*np.pi) - np.pi)
    displacement = np.linalg.norm(states[:, 2:4] - history[2:4], axis=1)
    outside = np.any((states[:, :4] < 40) | (states[:, :4] > 472), axis=1)
    return {"success_count": int(np.sum((pos < 20) & (angle < np.pi/9))),
            "best_joint_position_error": float(pos.min()),
            "median_joint_position_error": float(np.median(pos)),
            "median_object_displacement": float(np.median(displacement)),
            "terminal_center_outside_count": int(outside.sum()),
            "trajectory_center_outside_count": int(np.sum(crossed))}


def main():
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
    base = old.visible.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight failed")
    envpath = ROOT / runtime.ENV_REL
    spec = importlib.util.spec_from_file_location("proposal_scale_native", envpath)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    files = [PROTOCOL, Path(__file__), Path(old.__file__), envpath,
             *(DATA / n for n in ("states.pth", "velocities.pth", "rel_actions.pth"))]
    bindings = {str(p): sha(p) for p in files}
    states, velocities, rel = [torch.load(DATA / n, map_location="cpu", weights_only=True).numpy()
                               for n in ("states.pth", "velocities.pth", "rel_actions.pth")]
    proposed = banks()
    OUT.mkdir()
    rows = []
    for seq in (9, 11):
        initial = np.concatenate((states[seq, 0], velocities[seq, 0])).astype(np.float64)
        warm = (rel[seq, :10] / 100).astype(np.float32)
        reference = (rel[seq, 10:25] / 100).astype(np.float32)

        def execute(chain):
            env = native.PushTEnv(**dict(base.CTOR, reset_to_state=initial.copy()))
            try:
                env.seed(0)
                _, _ = env.reset()
                for a in warm:
                    _, _, done, info = env.step(a.astype(np.float64))
                    if done:
                        raise ValueError("warmup termination")
                hist = info["state"].copy()
                crossing = False
                for a in chain:
                    _, _, done, info = env.step(a.astype(np.float64))
                    if done:
                        raise ValueError("candidate termination")
                    current = info["state"]
                    crossing |= bool(np.any((current[:4] < 40) | (current[:4] > 472)))
                return hist, info["state"].copy(), crossing
            finally:
                env.close()

        history, goal, _ = execute(reference)
        for arm, bank in proposed.items():
            start = time.monotonic()
            finals, crossings = [], []
            for chain in bank:
                h, s, crossed = execute(chain)
                np.testing.assert_array_equal(h, history)
                finals.append(s)
                crossings.append(crossed)
            finals = np.stack(finals)
            crossings = np.asarray(crossings)
            path = OUT / f"seq{seq}_{arm}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, actions=bank, states=finals, crossed=crossings,
                                    history=history, goal=goal)
            row = {"sequence": seq, "arm": arm, "count": len(bank),
                   "data_sha256": sha(path), "elapsed_seconds": time.monotonic() - start,
                   **metrics(finals, goal, history, crossings)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if any(sha(p) != digest for p, digest in bindings.items()):
        raise ValueError("dependency changed during experiment")
    with (OUT / "report.json").open("x") as handle:
        json.dump({"status": "completed", "bindings": bindings, "rows": rows,
                   "limits": "two exposed tasks; physical fixed-bank diagnostic, not planner"}, handle,
                  indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    print("Comparing physical coverage and visibility of fixed action proposal distributions.", flush=True)
    main()
