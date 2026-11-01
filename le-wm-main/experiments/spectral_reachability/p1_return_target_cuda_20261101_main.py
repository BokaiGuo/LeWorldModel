"""Render fixed push-and-return tasks, verify renderer, encode on host CUDA."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from . import e3_cp10_target_cuda_20261031 as source

NAME = "p1_return_target_cuda_20261101_main"
DOC = source.ROOT.parent / "docs/D2P_P1_RETURN_TARGET_PROTOCOL_20261101_MAIN.md"


def return_goal(root_state, object_goal):
    root = np.asarray(root_state, np.float64)
    target = np.asarray(object_goal, np.float64).copy()
    if root.shape != (7,) or target.shape != (7,) or not np.isfinite([root, target]).all():
        raise ValueError("finite state7")
    target[:2], target[5:] = root[:2], root[5:]
    return target


def configure():
    original = source.collect

    def collect():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        from . import codex_four_arm_prefix_intervention_20261030 as physical
        entries, sources = original()
        witness = json.loads(source.REPORT.read_bytes())
        witness_rows = {row["root_slot"]: row for row in witness["rows"] if row["arm"] == "endpoints"}
        prepare = source.ROOT / "results/p1_prefix_protected_carry_20261101_main_root0_PREP_MAIN01.json"
        if source.base.sha(prepare) != "10b9a97b87b17e16ccfdc12f17fdf22292bdb51b675da0c83457bb7166ba24f2":
            raise ValueError("exact replay root source")
        prepared = json.loads(prepare.read_bytes())
        if any(source.base.sha(p) != h for p, h in prepared["sources"].items()):
            raise ValueError("replay source drift")
        data = physical.p3.source_data(prepared["source"])
        sources.update(prepared["sources"])
        sources[str(prepare)] = source.base.sha(prepare)
        output = []
        for entry in entries:
            env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                           with_target=False, render_action=False)
            try:
                old_goal = np.asarray(entry["goal_state"], np.float64)
                body = env.unwrapped
                ref = data["references"][entry["root"]]
                _, info = env.reset(seed=ref["slot"]["reset_seed"])
                original_goal = np.asarray(info["goal_state"], np.float64)
                requested = physical.moving.source.sample_state(
                    original_goal, list(body.block.shapes),
                    dict(root_seed=ref["slot"]["root_seed"]), 0, 0, data["plan"])
                obs, _ = env.reset(seed=0, options=dict(state=requested.copy(), goal_state=original_goal.copy()))
                history = [np.asarray(env.render()).copy()]
                states = [np.asarray(obs["state"]).copy()]
                for step, command in enumerate(ref["raw_history"], 1):
                    obs, _, terminated, truncated, _ = env.step(command.copy())
                    if terminated or truncated or not np.array_equal(body.latest_action, command):
                        raise ValueError("warmup replay stopped or action drift")
                    if step in (5, 10):
                        history.append(np.asarray(env.render()).copy())
                        states.append(np.asarray(obs["state"]).copy())
                if not np.array_equal(history, ref["history"]) or not np.array_equal(states, ref["states"]):
                    raise ValueError("exact replay root context")
                physical.moving.old.kernel.moving.attach_goal(body, ref["goal"], ref["goal_rgb"])
                commands = np.frombuffer(bytes.fromhex(entry["witness10_float32_hex"]), dtype="<f4").reshape(10, 2)
                for step, command in enumerate(commands, 1):
                    obs, _, terminated, truncated, _ = env.step(command.copy())
                    expected = witness_rows[entry["root"]]["responses"][step-1]
                    if (not np.array_equal(body.latest_action, command)
                            or not np.array_equal(obs["state"], expected["state"])
                            or bool(terminated) != expected["terminated"]
                            or bool(truncated) != expected["truncated"]):
                        raise ValueError("historical witness replay drift")
                if not np.array_equal(body._get_obs(), old_goal):
                    raise ValueError("original replay target state")
                old_pixels = np.asarray(env.render())
                if not np.array_equal(old_pixels, entry["pixels"]):
                    raise ValueError("native target renderer mismatch: root="
                                     f"{entry['root']} pixels="
                                     f"{np.any(old_pixels != entry['pixels'], axis=-1).sum()}")
                goal = return_goal(entry["root_state"], old_goal)
                body.agent.position, body.agent.velocity = tuple(goal[:2]), tuple(goal[5:])
                body.space.reindex_shapes_for_body(body.agent)
                pixels = np.asarray(env.render()).copy()
                if (not np.array_equal(body._get_obs(), goal) or pixels.shape != (224, 224, 3)
                        or pixels.dtype != np.uint8):
                    raise ValueError("rendered target state/image")
                native = source.qualification.native(goal, entry["root_state"])
                output.append(dict(root=entry["root"], root_state=entry["root_state"],
                                   goal_state=goal.tolist(), pixels=pixels,
                                   pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                   pixels_uint8_base64=base64.b64encode(pixels.tobytes()).decode(),
                                   native_initial=native, eligible_control=not native["initial_success"],
                                   source_goal_pixel_sha256=entry["pixel_sha256"],
                                   historical_action_witness=False,
                                   generation=dict(resets=2, public_steps=20,
                                                   raw_warmup=ref["raw_history"].tolist(),
                                                   witness_commands=commands.tolist(),
                                                   implicit_native_reset_physics_steps=4,
                                                   target_agent_placements=1, exact_original_pixels=True)))
            finally:
                env.close()
        for path in (DOC, Path(__file__).resolve()):
            sources[str(path)] = source.base.sha(path)
        return output, sources

    source.collect, source.NAME = collect, NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    configure()
    source.run(args.output)
