"""Physical-only replay of sealed20 programs on the fixed centered root."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_coherent_intervention_20261101_main as common


def execute(schedule, bindings, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    if not output.is_absolute() or output.parent != common.panel.ROOT / "results" or output.exists():
        raise ValueError("exclusive output")
    names = [s["prefix"] for s in schedule]
    if names != [*common.panel.ARMS, "zero"]:
        raise ValueError("five fixed prefixes")
    for s in schedule:
        raw = np.asarray(s["commands"], np.float32)
        if raw.shape != (20, 2) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
            raise ValueError("bounded sealed20")
    goal, images, sources = common.target.collect()
    root = json.loads((common.target.DIRECTORY / "report.json").read_bytes())
    sources.update(bindings)
    sources[str(Path(__file__).resolve())] = common.panel.saved.sha(Path(__file__))
    if any(common.panel.saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    output.mkdir()
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(schedule=schedule, sources=sources, neural_calls=0,
                       P1_admission=False, independent_confirmation=False), stream, indent=2)
    rows = []
    for slot in schedule:
        row = dict(prefix=slot["prefix"], status="failed", checkpoints=[], executed_raw=[])
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                       with_target=False, render_action=False)
        try:
            body = env.unwrapped
            env.reset(seed=root["reset_seed"])
            env.reset(seed=0, options=dict(state=np.asarray(root["requested_reset"]),
                                          goal_state=np.asarray(root["placeholder_goal"])))
            for step in range(11):
                if step:
                    action = np.asarray(root["raw_history"][step-1], np.float32)
                    _, _, stop, trunc, _ = env.step(action.copy())
                    if stop or trunc or not np.array_equal(body.latest_action, action):
                        raise ValueError("warmup stop/action mismatch")
                if step in (0, 5, 10) and not np.array_equal(env.render(), images[step//5]):
                    raise ValueError("exact warmup RGB")
            common.native_source.attach_goal(body, np.asarray(goal["goal_state"]), images[-1])

            def capture(step):
                state = np.asarray(body._get_obs()).copy()
                pixels = np.asarray(env.render()).copy()
                if state.shape != (7,) or not np.isfinite(state).all():
                    raise ValueError("finite physical state")
                path = output / f"{slot['prefix']}_{step:02}.png"
                Image.fromarray(pixels).save(path)
                native = common.metrics.qualification.native(goal["goal_state"], state)
                point = dict(step=step, state=state.tolist(),
                    metrics=dict(**common.metrics.metrics(state, goal["goal_state"]),
                                 joint_xy=native["joint_xy_error"], native_success=native["initial_success"]),
                    png=str(path), file_sha256=common.panel.saved.sha(path),
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
                row["checkpoints"].append(point)
                if step == 5 and (point["pixel_sha256"] != slot["cp5_pixel_sha256"]
                        or common.panel.saved.spec.digest(point["state"]) != slot["cp5_state_identity_sha256"]):
                    raise ValueError("actual prefix differs from planning input")
                return state

            if not np.array_equal(capture(0), goal["root_state"]):
                raise ValueError("shared root")
            for step, command in enumerate(np.asarray(slot["commands"], np.float32), 1):
                obs, _, stop, trunc, info = env.step(command.copy())
                if not np.array_equal(body.latest_action, command) or not np.array_equal(info["goal_state"], goal["goal_state"]):
                    raise ValueError("executed action or goal")
                row["executed_raw"].append(command.tolist())
                state = np.asarray(obs["state"])
                outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
                if step % 5 == 0 or stop or trunc or outside:
                    capture(step)
                if stop or trunc or outside:
                    row.update(status="native_stopped" if stop or trunc else "canvas_censored",
                               terminated=bool(stop), truncated=bool(trunc))
                    break
            else:
                row["status"] = "completed20"
        except Exception as exc:
            row["error"] = dict(type=type(exc).__name__, message=str(exc))
        finally:
            env.close()
        row["steps"] = len(row["executed_raw"])
        rows.append(row)
        with (output / f"{slot['prefix']}.json").open("x") as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
        print(json.dumps({k: row.get(k) for k in ("prefix", "status", "steps", "error")}), flush=True)
    if any(common.panel.saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    with (output / "report.json").open("x") as stream:
        json.dump(dict(rows=rows, sources=sources,
                       protocol_sha256=common.panel.saved.sha(output / "protocol.json"),
                       neural_calls=0, P1_admission=False, independent_confirmation=False),
                  stream, indent=2, allow_nan=False)
    return all(r["status"] != "failed" for r in rows)
