"""Same-backend physical replay of both sealed feedback trajectories."""
import importlib.util
import json
import os
from pathlib import Path
import traceback

import numpy as np

from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior
from experiments.pc_jepa_native_v11_feasible_feedback_audit_20261104_codex import OUT, ROOT, sha


def main():
    print("Replaying both saved native trajectories; no planner or neural calls.", flush=True)
    destination = OUT / "physical_replay.json"
    if destination.exists():
        raise FileExistsError(destination)
    overlay = ROOT / ".jepa-physics-site"
    if os.environ.get("PYTHONPATH") != str(overlay):
        raise ValueError("physics overlay required")
    base = prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")
    if probe["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime probe failed")
    spec = importlib.util.spec_from_file_location("pc_jepa_feedback_replay_native", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manifest = json.loads((OUT / "manifest.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    if audit["status"] != "saved_data_audit_passed" or audit["summary_sha256"] != sha(OUT / "summary.json"):
        raise ValueError("saved audit not bound")
    results = {}
    for arm in ("response", "cem"):
        env, error, rows = None, None, []
        try:
            env = native.PushTEnv(**base.CTOR)
            env.seed(manifest["seed"])
            obs, state = env.reset()
            path = OUT / f"{arm}_physical"
            def check(frame, obs, state, transition):
                frame_path = path / f"frame_{frame:02d}"
                checks = {
                    "rgb": bool(np.array_equal(obs["visual"], np.load(frame_path / "rgb.npy", allow_pickle=False))),
                    "state": bool(np.array_equal(state, np.load(frame_path / "state.npy", allow_pickle=False))),
                    "proprio": bool(np.array_equal(obs["proprio"], np.load(frame_path / "proprio.npy", allow_pickle=False))),
                    "public": base.public_state(env) == json.loads((frame_path / "full_public_state.json").read_text()),
                    "transition": base.numeric_json(transition) == json.loads((frame_path / "transition.json").read_text())}
                rows.append({"frame": frame, "exact": checks})
                if not all(checks.values()):
                    raise ValueError(f"frame {frame} mismatch: {checks}")
            check(0, obs, state, {"kind": "reset"})
            for i in range(1, 26):
                u = np.asarray(manifest["warmup"][i - 1] if i <= 10 else
                               json.loads((path / f"command_{i:02d}.json").read_text())["action"], np.float64)
                obs, reward, done, info = env.step(u)
                check(i, obs, info["state"], {"kind": "step", "reward": reward, "done": done, "info": info})
                if done:
                    raise ValueError("unexpected done")
        except Exception:
            error = traceback.format_exc()
        finally:
            if env is not None:
                env.close()
        results[arm] = {"status": "exact" if error is None and len(rows) == 26 else "failed",
                        "frames": rows, "error": error}
    record = {"status": "exact" if all(v["status"] == "exact" for v in results.values()) else "failed",
              "manifest_sha256": sha(OUT / "manifest.json"), "summary_sha256": sha(OUT / "summary.json"),
              "audit_sha256": sha(OUT / "audit.json"), "arms": results,
              "scope": "same native backend physical replay; private solver caches are not independently captured"}
    with destination.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v["status"] for k, v in results.items()}), flush=True)
    if record["status"] != "exact":
        raise RuntimeError("physical replay failed; retained")


if __name__ == "__main__":
    main()
