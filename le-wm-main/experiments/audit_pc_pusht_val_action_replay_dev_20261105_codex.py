"""Development replay from partial PushT dataset states; no planning."""
import hashlib
import importlib.util
import json
import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments import pc_jepa_visible_local_bank_dev_20261105_codex as existing

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/jepa_wm_osf/extracted/pusht_noise/val"
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_VAL_ACTION_REPLAY_DEV_PROTOCOL_20261105_CODEX.md"
OUT = ROOT / "results/pc_pusht_val_action_replay_dev_20261105_codex.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def errors(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return {"agent_position": float(np.linalg.norm(actual[:2] - reference[:2])),
            "object_position": float(np.linalg.norm(actual[2:4] - reference[2:4])),
            "angle": float(abs((actual[4] - reference[4] + np.pi) % (2*np.pi) - np.pi)),
            "agent_velocity": float(np.linalg.norm(actual[5:7] - reference[5:7]))}


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
    base = existing.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("val_action_replay_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    names = ("states.pth", "velocities.pth", "rel_actions.pth", "abs_actions.pth", "seq_lengths.pkl")
    bindings = {name: sha(DATA / name) for name in names}
    arrays = [torch.load(DATA / name, map_location="cpu", weights_only=True).numpy()
              for name in names[:-1]]
    states, velocities, rel, absolute = arrays
    with (DATA / "seq_lengths.pkl").open("rb") as f:
        lengths = pickle.load(f)
    if states.shape[-1] != 5 or velocities.shape[-1] != 2 or rel.shape[-1] != 2 or absolute.shape != rel.shape:
        raise ValueError("unexpected dataset dimensions")
    rows = []
    for i in (9, 11):
        length = int(lengths[i])
        if length < 26:
            raise ValueError(f"sequence {i} too short")
        reference = np.concatenate((states[i], velocities[i]), axis=-1).astype(np.float64)
        delta = absolute[i, :length].astype(np.float64) - states[i, :length, :2] - rel[i, :length]
        entry = {"sequence": i, "length": length,
                 "absolute_minus_position_minus_relative_abs_max": float(np.max(np.abs(delta))),
                 "absolute_minus_position_minus_relative_abs_quantiles": np.quantile(np.abs(delta), [.5, .95, 1]).tolist(),
                 "trials": []}
        rows.append(entry)
        trials = [("open_from_0", 0, 25), ("open_from_10", 10, 25)]
        trials += [(f"one_step_{t}", t, t + 1) for t in range(10, 25)]
        for name, start, stop in trials:
            trial = {"name": name, "start": start, "stop": stop, "status": "running", "frames": []}
            entry["trials"].append(trial)
            env = None
            try:
                ctor = dict(base.CTOR, reset_to_state=reference[start].copy())
                env = native.PushTEnv(**ctor)
                env.seed(0)
                _, actual = env.reset()
                trial["reset_error"] = errors(actual, reference[start])
                for t in range(start, stop):
                    _, _, done, info = env.step(rel[i, t].astype(np.float64) / 100)
                    if done:
                        raise RuntimeError(f"early done at {t}")
                    actual = np.asarray(info["state"], dtype=np.float64)
                    trial["frames"].append({"target_index": t+1, "actual": actual.tolist(),
                                             "reference": reference[t+1].tolist(),
                                             "errors": errors(actual, reference[t+1])})
                trial["status"] = "completed"
            except Exception:
                trial["status"], trial["error"] = "failed", traceback.format_exc()
            finally:
                if env is not None:
                    env.close()
        print(f"sequence {i}: {len(entry['trials'])} replay trials", flush=True)
    if any(sha(DATA / name) != digest for name, digest in bindings.items()):
        raise ValueError("data changed during replay")
    output = {"status": "completed" if all(t["status"] == "completed" for row in rows for t in row["trials"]) else "failed",
              "scope": "partial recorded state and one same-backend replay; no checkpoint or data provenance",
              "data_sha256": bindings, "env_sha256": sha(ROOT / runtime.ENV_REL),
              "protocol_sha256": sha(PROTOCOL), "auditor_sha256": sha(Path(__file__)), "rows": rows}
    with OUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")
    if output["status"] != "completed":
        raise RuntimeError("one or more replay trials failed; see result")


if __name__ == "__main__":
    main()
