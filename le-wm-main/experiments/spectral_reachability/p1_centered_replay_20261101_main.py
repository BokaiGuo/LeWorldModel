"""Replay the centered witness under its actual native goal, honoring stops."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_centered_target_cuda_20261101_main as target
from . import p1_constructive_moving_history_source_20261027_v1 as native_source


def run(output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    entry, images, sources = target.collect()
    directory = target.DIRECTORY
    report = json.loads((directory / "report.json").read_bytes())
    out = Path(output)
    if not out.is_absolute() or out.parent != directory.parent or out.exists():
        raise ValueError("exclusive output")
    path = Path(__file__).resolve()
    sources[str(path)] = target.encoder.base.sha(path)
    path = Path(native_source.__file__).resolve()
    sources[str(path)] = target.encoder.base.sha(path)
    result = dict(status="failed", sources=sources, rows=[], neural_calls=0,
                  P1_admission=False, independent_confirmation=False)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                   with_target=False, render_action=False)
    try:
        body = env.unwrapped
        env.reset(seed=report["reset_seed"])
        env.reset(seed=0, options=dict(state=np.asarray(report["requested_reset"]),
                                      goal_state=np.asarray(report["placeholder_goal"])))
        for step in range(11):
            if step:
                action = np.asarray(report["raw_history"][step-1], np.float32)
                _, _, stop, trunc, _ = env.step(action.copy())
                if stop or trunc or not np.array_equal(body.latest_action, action):
                    raise ValueError("warmup stop/action mismatch")
            if step in (0, 5, 10):
                if not np.array_equal(env.render(), images[step//5]):
                    raise ValueError("exact RGB warmup replay")
        native_source.attach_goal(body, np.asarray(entry["goal_state"]), images[-1])
        if not np.array_equal(env.render(), images[2]):
            raise ValueError("goal attachment changed root RGB")
        for row in report["rows"][1:]:
            action = np.asarray(row["command"], np.float32)
            obs, reward, stop, trunc, info = env.step(action.copy())
            state = np.asarray(obs["state"])
            pixels = np.asarray(env.render())
            if (not np.array_equal(body.latest_action, action)
                    or not np.array_equal(state, row["state"])
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != row["pixel_sha256"]
                    or not np.array_equal(info["goal_state"], entry["goal_state"])):
                raise ValueError("same-action native-goal replay identity")
            native = target.witness.old.qualification.native(entry["goal_state"], state)
            result["rows"].append(dict(step=row["step"], state=state.tolist(),
                                       pixel_sha256=row["pixel_sha256"], reward=float(reward),
                                       terminated=bool(stop), truncated=bool(trunc), native=native))
            if stop or trunc:
                break
        result.update(status="replayed", native_success=result["rows"][-1]["native"]["initial_success"],
                      steps=len(result["rows"]), note="scripted witness, not model-selected method")
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        env.close()
        if any(target.encoder.base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source drift")
        with out.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result.get(k) for k in ("status", "native_success", "steps", "error")}), flush=True)
    return result["status"] == "replayed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", required=True, action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
