"""Instrument the local native evaluator without changing its planning policy."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import hydra
import stable_worldmodel as swm

from eval import run


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating):
        return np.where(np.isfinite(value), value, None).tolist()
    return value.tolist()


original_get_actions = swm.World._get_actions
original_run = swm.World._run


def traced_get_actions(world):
    actions = original_get_actions(world)
    world._pc_trace_last_action = np.asarray(actions).copy()
    return actions


def traced_run(world, *args, **kwargs):
    trace = {
        "initial_state": as_list(world.infos["state"]),
        "goal_state": as_list(world.infos["goal_state"]),
        "steps": [],
    }
    old_callback = kwargs.get("on_step")
    alive = np.ones(world.num_envs, dtype=bool)

    def callback(current):
        if old_callback is not None:
            old_callback(current)
        trace["steps"].append({
            "action": as_list(current._pc_trace_last_action),
            "state": as_list(current.infos["state"]),
            "active_before_step": alive.tolist(),
            "terminated": as_list(current.terminateds),
            "truncated": as_list(current.truncateds),
        })
        alive[:] &= ~(current.terminateds | current.truncateds)

    kwargs["on_step"] = callback
    try:
        return original_run(world, *args, **kwargs)
    finally:
        with Path(os.environ["PC_NATIVE_TRACE_PATH"]).open("x") as stream:
            json.dump(trace, stream, allow_nan=False)
            stream.write("\n")


swm.World._get_actions = traced_get_actions
swm.World._run = traced_run


@hydra.main(version_base=None, config_path=str(Path(__file__).resolve().parents[1] / "config/eval"), config_name="pusht")
def traced_eval(cfg):
    return run.__wrapped__(cfg)


if __name__ == "__main__":
    traced_eval()
