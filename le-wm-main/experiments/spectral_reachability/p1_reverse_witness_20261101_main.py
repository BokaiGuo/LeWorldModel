"""Generate one prospective20-step opposite-side push witness, without a model."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_arm_prefix_intervention_20261030 as physical
from . import e3_initial_goal_qualification_20261031 as qualification

NAME = "p1_reverse_witness_20261101_main"
ROOT = physical.ROOT
DOC = ROOT.parent / "docs/D2P_P1_REVERSE_WITNESS_PROTOCOL_20261101_MAIN.md"
PREP = ROOT / "results/p1_prefix_protected_carry_20261101_main_root0_PREP_MAIN01.json"
PREP_SHA = "10b9a97b87b17e16ccfdc12f17fdf22292bdb51b675da0c83457bb7166ba24f2"
PHASES = ((130, -100, 4), (-100, -110, 6), (-70, -70, 3), (45, 45, 4), (130, 80, 3))


def eligibility(states):
    if len(states) != 21:
        return dict(complete20=False)
    initial, prefix, target = (np.asarray(states[i], np.float64) for i in (0, 5, 20))
    move = target[2:4] - initial[2:4]
    direction = initial[:2] - initial[2:4]
    before = qualification.native(target, initial)
    after = qualification.native(target, prefix)
    return dict(complete20=True, initially_unsuccessful=not before["initial_success"],
                object_moves_at_least20=float(np.linalg.norm(move)) >= 20,
                object_moves_toward_initial_pusher=float(move @ direction) > 0,
                cp5_object_unchanged=bool(np.max(np.abs(prefix[2:5] - initial[2:5])) <= 1e-8),
                cp5_joint_xy_nonimproving=after["joint_xy_error"] >= before["joint_xy_error"])


def run(output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    if (not output.is_absolute() or output.parent != ROOT / "results" or output.exists()
            or not output.name.startswith(NAME)):
        raise ValueError("exclusive output directory")
    if physical.sha(PREP) != PREP_SHA:
        raise ValueError("exact root prepare")
    prepared = json.loads(PREP.read_bytes())
    if any(physical.sha(p) != digest for p, digest in prepared["sources"].items()):
        raise ValueError("root source drift")
    data = physical.p3.source_data(prepared["source"])
    ref = data["references"][0]
    sources = dict(prepared["sources"])
    for path in (PREP, DOC, Path(__file__).resolve()):
        sources[str(path)] = physical.sha(path)
    output.mkdir()
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(phases=PHASES, sources=sources, role="privileged_task_generator_only",
                       neural_calls=0), stream, indent=2)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                   with_target=False, render_action=False)
    report = dict(status="partial_or_failed", rows=[], sources=sources, neural_calls=0,
                  P1_admission=False, independent_confirmation=False, task_eligible=False)
    try:
        body = env.unwrapped
        _, info = env.reset(seed=ref["slot"]["reset_seed"])
        original_goal = np.asarray(info["goal_state"], np.float64)
        requested = physical.moving.source.sample_state(
            original_goal, list(body.block.shapes), dict(root_seed=ref["slot"]["root_seed"]),
            0, 0, data["plan"])
        obs, _ = env.reset(seed=0, options=dict(state=requested.copy(), goal_state=original_goal.copy()))
        history, states = [np.asarray(env.render()).copy()], [np.asarray(obs["state"]).copy()]
        for step, action in enumerate(ref["raw_history"], 1):
            obs, _, stopped, truncated, _ = env.step(action.copy())
            if stopped or truncated or not np.array_equal(body.latest_action, action):
                raise ValueError("warmup action/stop")
            if step in (5, 10):
                history.append(np.asarray(env.render()).copy())
                states.append(np.asarray(obs["state"]).copy())
        if not np.array_equal(history, ref["history"]) or not np.array_equal(states, ref["states"]):
            raise ValueError("exact reset/warmup context")

        def capture(step, action=None, terminated=False, truncated=False):
            pixels = np.asarray(env.render()).copy()
            state = np.asarray(body._get_obs()).copy()
            if state.shape != (7,) or not np.isfinite(state).all():
                raise ValueError("finite physical state")
            path = output / f"step_{step:02}.png"
            Image.fromarray(pixels).save(path)
            item = dict(step=step, state=state.tolist(), command=None if action is None else action.tolist(),
                        terminated=bool(terminated), truncated=bool(truncated), png=str(path),
                        file_sha256=physical.sha(path),
                        pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
            report["rows"].append(item)
            return state, pixels

        initial, _ = capture(0)
        center = initial[2:4].copy()
        halted = False
        for dx, dy, count in PHASES:
            waypoint = center + np.array([dx, dy])
            for _ in range(count):
                current = np.asarray(body.agent.position)
                command = np.clip((waypoint - current) / 100., -1, 1).astype(np.float32)
                obs, _, terminated, truncated, _ = env.step(command.copy())
                if not np.array_equal(body.latest_action, command) or not np.array_equal(obs["state"], body._get_obs()):
                    raise ValueError("public action/state identity")
                state, pixels = capture(len(report["rows"]), command, terminated, truncated)
                if not bool(((state[:4] >= 0) & (state[:4] <= 512)).all()):
                    report["domain_censored"] = report["rows"][-1]["step"]
                    halted = True
                    break
                if terminated or truncated:
                    report["native_stop"] = report["rows"][-1]["step"]
                    halted = True
                    break
            if halted:
                break
        checks = eligibility([r["state"] for r in report["rows"]])
        eligible = all(checks.values()) and not halted
        report.update(status="witness_recorded", checks=checks, task_eligible=eligible,
                      public_control_steps=len(report["rows"]) - 1, exact_root=True,
                      target=dict(state=state.tolist(),
                                  pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                  pixels_uint8_base64=base64.b64encode(pixels.tobytes()).decode())
                      if checks["complete20"] else None)
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        env.close()
        if any(physical.sha(p) != digest for p, digest in sources.items()):
            raise ValueError("source drift")
        with (output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report["status"], task_eligible=report["task_eligible"],
                          checks=report.get("checks"), error=report.get("error"),
                          report_sha256=physical.sha(output / "report.json"))), flush=True)
    return report["status"] == "witness_recorded"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
