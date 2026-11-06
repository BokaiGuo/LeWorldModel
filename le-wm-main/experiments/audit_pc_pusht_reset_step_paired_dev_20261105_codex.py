"""Compare standard reset with an evaluation-only agent state correction."""
import importlib.util
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments import audit_pc_pusht_val_action_replay_dev_20261105_codex as prior

ROOT = prior.ROOT
DATA = prior.DATA
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_RESET_STEP_PAIRED_DEV_PROTOCOL_20261105_CODEX.md"
OUT = ROOT / "results/pc_pusht_reset_step_paired_dev_20261105_codex.json"


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
    base = prior.existing.core.old.old.prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != \
            "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight")
    spec = importlib.util.spec_from_file_location("reset_pair_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    names = ("states.pth", "velocities.pth", "rel_actions.pth", "seq_lengths.pkl")
    hashes = {name: prior.sha(DATA / name) for name in names}
    states, vel, actions = [torch.load(DATA / name, map_location="cpu", weights_only=True).numpy()
                            for name in names[:3]]
    with (DATA / names[3]).open("rb") as f:
        lengths = pickle.load(f)
    rows = []
    for sequence in (9, 11):
        if lengths[sequence] < 26:
            raise ValueError("short sequence")
        ref = np.concatenate((states[sequence], vel[sequence]), axis=-1).astype(np.float64)
        for t in range(10, 25):
            pair = {"sequence": sequence, "time": t}
            for mode in ("standard", "agent_state_corrected"):
                env = native.PushTEnv(**dict(base.CTOR, reset_to_state=ref[t].copy()))
                try:
                    env.seed(0)
                    env.reset()
                    if mode == "agent_state_corrected":
                        env.agent.position = tuple(ref[t, :2])
                        env.agent.velocity = tuple(ref[t, 5:7])
                        env.space.reindex_shapes_for_body(env.agent)
                    reset = prior.errors(env._get_obs(), ref[t])
                    _, _, done, info = env.step(actions[sequence, t].astype(np.float64) / 100)
                    if done:
                        raise RuntimeError("unexpected early done")
                    pair[mode] = {"reset_error": reset, "next_error": prior.errors(info["state"], ref[t+1])}
                finally:
                    env.close()
            rows.append(pair)
        print(f"sequence {sequence}: 15 paired trials", flush=True)
    if any(prior.sha(DATA / name) != digest for name, digest in hashes.items()):
        raise ValueError("data changed")
    output = {"status": "completed", "protocol_sha256": prior.sha(PROTOCOL), "auditor_sha256": prior.sha(Path(__file__)),
              "dependency_sha256": prior.sha(Path(prior.__file__)), "env_sha256": prior.sha(ROOT / runtime.ENV_REL),
              "data_sha256": hashes, "rows": rows,
              "scope": "evaluation-only intervention; no complete object dynamics restore"}
    with OUT.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
