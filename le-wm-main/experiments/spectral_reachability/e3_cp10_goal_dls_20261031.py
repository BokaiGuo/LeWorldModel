"""Known-root exploratory goal-conditioned replacement of the spectral bank."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import e3_cp10_structured_extra32_20261031 as prior
from . import p1_four_arm_extra32_budget_20261031 as control

NAME = "e3_cp10_goal_dls_20261031"
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/D2P_E3_GOAL_DLS_INITIALIZER_PROTOCOL_20261031.md"
SCALES = np.array([-1., -.5, -.25, 0., .25, .5, .75, 1.])


def goal_bank(jac, projection, goal, nominal):
    jac = np.asarray(jac, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    nominal = np.asarray(nominal, dtype=np.float64)
    if (jac.shape != (192, 30) or projection.shape != (192, 192)
            or goal.shape != (192,) or nominal.shape != (192,)
            or not all(np.isfinite(x).all() for x in (jac, projection, goal, nominal))):
        raise ValueError("invalid DLS inputs")
    matrix = projection @ jac
    residual = projection @ (goal - nominal)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] == 0:
        direction = np.zeros(30)
    else:
        damping = .01 * singular[0] ** 2
        direction = right.T @ ((singular / (singular**2 + damping)) * (left.T @ residual))
        direction /= max(1., np.max(np.abs(direction)))
    bank = (SCALES[:, None] * direction).reshape(8, 3, 5, 2).astype(np.float32)
    if not np.isfinite(bank).all() or np.max(np.abs(bank)) > 1:
        raise ValueError("invalid DLS bank")
    return bank, singular


def setup(root):
    experiment = prior.setup(root)
    runner = experiment.runner
    old_build, old_protocol, original_plan = runner.build, runner.protocol, control.plan

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), PROTOCOL):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(initializer="goal_conditioned_dls_8", dls_relative_damping=.01,
                     dls_scales=SCALES.tolist(), independent_confirmation=False)
        return value

    def plan(leaf, arm, task, config, policy, contract, *, device, emit):
        signal = {}

        def capture(kind, payload):
            if kind == "spectral_signal":
                signal["nominal"] = np.asarray(payload["nominal"], dtype=np.float64).reshape(192)
            emit(kind, payload)

        def bank(jac, projection):
            if "nominal" not in signal:
                raise ValueError("nominal prediction was not emitted")
            return goal_bank(jac, projection, task["goal"], signal["nominal"])

        with patch.object(control.base.spectral, "spectral_bank", bank):
            return original_plan(leaf, arm, task, config, policy, contract, device=device, emit=capture)

    control.plan = plan
    runner.NAME = f"{NAME}_root{root}"
    runner.build, runner.protocol = build, protocol
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    experiment = setup(args.root)
    runner = experiment.runner
    if args.command == "prepare":
        result = runner.prepare(args.output)
    else:
        print("Exploratory goal-conditioned initializer; host CUDA and physical feedback.", flush=True)
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
