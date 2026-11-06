"""Generate one bounded evaluator-only native contact witness, then replay it."""
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np

from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_jepa_v22_new_contact_witness_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_NEW_CONTACT_WITNESS_PROTOCOL_20261104_CODEX.md"
SEED = 2026110451


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics-only overlay required")
    base = prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")
    if probe["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime probe failed")
    env_path = ROOT / runtime.ENV_REL
    spec = importlib.util.spec_from_file_location("pc_jepa_contact_witness_native", env_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    deps = seal.capture([Path(__file__), PROTOCOL, env_path, Path(prior.__file__),
                         ROOT / f"results/{base.RUNTIME}/manifest_frozen.json"])
    OUT.mkdir()
    warm = np.asarray(prior.SCHEDULE, dtype=np.float64).reshape(15, 2)[:10]
    base.save_json(OUT / "manifest.json", {"seed": SEED, "warmup": warm.tolist(),
        "actions": 15, "policy": "float32(clip((object_xy-agent_xy)/100,-1,1))",
        "dependencies": deps, "scope": "privileged evaluation-only constructed witness"})
    rows, error = [], None
    actions = []
    try:
        for run in ("construct", "replay"):
            env = native.PushTEnv(**base.CTOR)
            try:
                env.seed(SEED)
                obs, state = env.reset()
                rgbs, props, states = [obs["visual"].copy()], [obs["proprio"].copy()], [state.copy()]
                for index in range(25):
                    if index < 10:
                        action = warm[index]
                    elif run == "construct":
                        action = np.clip((np.asarray(state[2:4], float) - np.asarray(state[:2], float)) / 100,
                                         -1, 1).astype(np.float32)
                        actions.append(action.copy())
                    else:
                        action = actions[index - 10]
                    if not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
                        raise ValueError("illegal action")
                    obs, _, done, info = env.step(np.asarray(action, np.float64))
                    if done:
                        raise ValueError("environment terminated")
                    state = info["state"]
                    rgbs.append(obs["visual"].copy())
                    props.append(obs["proprio"].copy())
                    states.append(state.copy())
                rows.append({"rgb": np.stack(rgbs), "proprio": np.stack(props), "state": np.stack(states)})
            finally:
                env.close()
        for name in ("rgb", "proprio", "state"):
            np.testing.assert_array_equal(rows[0][name], rows[1][name])
        start, end = rows[0]["state"][10], rows[0]["state"][25]
        displacement = float(np.linalg.norm(end[2:4] - start[2:4]))
        angle = float(abs((end[4] - start[4] + np.pi) % (2*np.pi) - np.pi))
        nontrivial = bool(displacement > 20 or angle > np.pi / 9)
        with (OUT / "witness.npz").open("xb") as f:
            np.savez_compressed(f, actions=np.asarray(actions, np.float32), **rows[0])
        seal.verify(deps)
        summary = {"status": "completed", "exact_replay": True, "nontrivial": nontrivial,
                   "object_displacement": displacement, "angle_change_radians": angle,
                   "postwarm_state": start.tolist(), "goal_state_evaluation_only": end.tolist(),
                   "witness_sha256": seal.digest(OUT / "witness.npz"), "error": None}
    except Exception:
        error = traceback.format_exc()
        summary = {"status": "failed", "error": error, "completed_trajectories": len(rows),
                   "actions_generated": len(actions)}
    base.save_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if error:
        raise RuntimeError("witness failed; retained")


if __name__ == "__main__":
    main()
