"""Fixed translated development task; preserve true reset/warmup RGB."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_reverse_witness_20261101_main as old
from . import p1_rgb_visibility_20261101_main as visibility

NAME = "p1_centered_witness_20261101_main"
DOC = old.ROOT.parent / "docs/D2P_P1_CENTERED_PROTOCOL_20261101_MAIN.md"
OFFSET = (-80., -100.)
PHASES = ((130, -100, 4), (-100, -110, 6), (-70, -70, 3),
          (0, 70, 5), (130, 80, 2))


def translated(state):
    result = np.asarray(state, np.float64).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError("finite native7 state")
    result[:2] += OFFSET
    result[2:4] += OFFSET
    return result


def run(output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    if (not output.is_absolute() or output.parent != old.ROOT / "results"
            or output.exists() or not output.name.startswith(NAME)):
        raise ValueError("exclusive output")
    if old.physical.sha(old.PREP) != old.PREP_SHA:
        raise ValueError("sealed source prepare")
    prepared = json.loads(old.PREP.read_bytes())
    sources = dict(prepared["sources"])
    for path in (old.PREP, DOC, Path(__file__).resolve(),
                 Path(old.__file__).resolve(), Path(visibility.__file__).resolve()):
        sources[str(path)] = old.physical.sha(path)
    if any(old.physical.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    data = old.physical.p3.source_data(prepared["source"])
    ref = data["references"][0]
    output.mkdir()
    protocol = dict(offset=OFFSET, phases=PHASES, sources=sources,
                    role="privileged_development_task_generation", neural_calls=0)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    report = dict(status="failed", task_admitted=False, P1_admission=False,
                  independent_confirmation=False, sources=sources, rows=[], history=[],
                  neural_calls=0, reset_seed=ref["slot"]["reset_seed"])
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                   with_target=False, render_action=False)
    try:
        body = env.unwrapped
        _, info = env.reset(seed=ref["slot"]["reset_seed"])
        goal = translated(info["goal_state"])
        reset = translated(ref["states"][0])
        obs, _ = env.reset(seed=0, options=dict(state=reset.copy(), goal_state=goal.copy()))
        report.update(requested_reset=reset.tolist(), placeholder_goal=goal.tolist(),
                      raw_history=np.asarray(ref["raw_history"]).tolist())

        def capture(kind, step):
            pixels = np.asarray(env.render()).copy()
            state = np.asarray(body._get_obs()).copy()
            path = output / f"{kind}_{step:02}.png"
            Image.fromarray(pixels).save(path)
            return dict(step=step, state=state.tolist(), png=str(path),
                        file_sha256=old.physical.sha(path),
                        pixel_sha256=old.hashlib.sha256(pixels.tobytes()).hexdigest(),
                        visibility=visibility.inspect(pixels))

        report["history"].append(capture("warmup", 0))
        for step, action in enumerate(ref["raw_history"], 1):
            obs, _, stopped, truncated, _ = env.step(action.copy())
            if stopped or truncated or not np.array_equal(body.latest_action, action):
                raise ValueError("warmup native stop/action mismatch")
            if step in (5, 10):
                report["history"].append(capture("warmup", step))
        for actual, expected in zip(report["history"], ref["states"], strict=True):
            if not np.allclose(actual["state"], translated(expected), atol=1e-7, rtol=0):
                raise ValueError("translation did not preserve warmup dynamics")
        report["warmup_translation_verified"] = True
        report["rows"].append(capture("step", 0))
        center = np.asarray(body._get_obs())[2:4].copy()
        halted = False
        for dx, dy, count in PHASES:
            waypoint = center + (dx, dy)
            for _ in range(count):
                command = np.clip((waypoint - np.asarray(body.agent.position))/100,
                                  -1, 1).astype(np.float32)
                obs, _, stopped, truncated, _ = env.step(command.copy())
                if not np.array_equal(body.latest_action, command):
                    raise ValueError("actual action mismatch")
                row = capture("step", len(report["rows"]))
                row.update(command=command.tolist(), terminated=bool(stopped),
                           truncated=bool(truncated))
                report["rows"].append(row)
                in_canvas = bool(np.all((np.asarray(row["state"])[:4] >= 0)
                                       & (np.asarray(row["state"])[:4] <= 512)))
                if stopped or truncated or not in_canvas:
                    halted = True
                    break
            if halted:
                break
        checks = old.eligibility([r["state"] for r in report["rows"]])
        checks["all_frames_interior"] = all(r["visibility"]["eligible"]
                                           for r in report["history"] + report["rows"])
        checks["no_native_stop_or_canvas_censoring"] = not halted
        report.update(status="recorded", checks=checks, task_admitted=all(checks.values()))
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        env.close()
        if any(old.physical.sha(p) != h for p, h in sources.items()):
            raise ValueError("source drift")
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: report.get(k) for k in ("status", "checks", "error", "task_admitted")}),
          flush=True)
    return report["status"] == "recorded"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", required=True, action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
