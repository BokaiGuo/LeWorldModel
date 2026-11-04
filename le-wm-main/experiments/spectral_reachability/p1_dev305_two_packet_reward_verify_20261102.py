"""Independently recompute native reward and success from saved physical states."""
import argparse
import importlib
import inspect
from pathlib import Path

import numpy as np

from . import p1_dev305_two_packet_metric_decompose_20261102 as paired

base, need, ROOT = paired.base, paired.need, paired.ROOT
NAME = "p1_dev305_two_packet_reward_verify_20261102"
DECOMPOSE_SHA = "b7a40a07e7e423c651e31a31d258318dbcf53210763ce6dede49c7b1e4f5a561"


def analyze():
    physics = base.read(ROOT / "results" / f"{paired.actual.physical.source.NAME}_RUN_DEV01/report.json",
                        paired.actual.physical.saved.REPORT_SHA)
    capture = base.read(ROOT / "results" / "p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        paired.CAPTURE_SHA)
    summary = base.read(ROOT / "results" / "p1_dev305_two_packet_metric_decompose_20261102_DEV01.json",
                        DECOMPOSE_SHA)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    native = paired.actual.physical.source.base.base
    need(base.sha(inspect.getsourcefile(cls)) == native.native.ENV_SHA, "native environment sha")
    goal = np.asarray(capture["goal_state_evaluation_only"], dtype=float)
    errors, recomputed_success = [], 0
    for row in physics["rows"]:
        need(row["status"] == "fixed_horizon_completed", "only complete outcomes")
        for transition in row["transitions"]:
            state = np.asarray(transition["state"], dtype=float)
            distance = float(np.linalg.norm(goal - state))
            errors.append(abs(transition["reward"] + distance))
            angle = abs(float(goal[4] - state[4]))
            angle = min(angle, 2 * np.pi - angle)
            native_success = bool(np.linalg.norm(goal[:4] - state[:4]) < 20 and angle < np.pi / 9)
            need(native_success == transition["terminated"] and not transition["truncated"], "native success")
            recomputed_success += int(native_success)
    need(max(errors) <= 1e-9, "native reward formula")
    return {"schema": NAME + ".v1", "physical_sequences": len(physics["rows"]),
            "transitions": len(errors), "max_reward_absolute_error": max(errors),
            "native_success_transitions": recomputed_success,
            "native_reward": "negative L2 of seven-dimensional state including agent, object, angle, velocity",
            "native_success": "joint 4D agent/object position norm < 20 and angle difference < pi/9",
            "paired_summary_sha256": DECOMPOSE_SHA, "summary_rows": len(summary["rows"]),
            "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
