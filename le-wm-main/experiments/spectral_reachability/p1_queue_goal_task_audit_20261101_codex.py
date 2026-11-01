"""Read-only development task audit and non-neural H2 compiler fixture."""
import argparse
from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from . import p1_queue_goal_blind_cuda_20261101_codex as run
from . import p1_conditional_task_admission_20261027_v1 as core
from . import p1_conditional_direct_screen_20261027_v1 as screen
from . import p3_goal_patch_consistent_compiler_20261027_v1 as compiler

ROOT = run.ROOT
PREPARED = ROOT / "results/p1_queue_goal_blind_cuda_20261101_codex_PREPARE_DEV01.json"
PREPARE_SHA = "1ac248ec2c3c7ee0715f0e37d3a8de670f1a556411754dcebe95c3caaec837e3"
RESULT = ROOT / "results/p1_queue_goal_blind_cuda_20261101_codex_RUN_DEV01"
REPORT_SHA = "d7aaea46ce7fd00ac5be2ee4daebfcc7c185cec5fbfc12fc8555a861b97ddbec"
TASK_SHA = "c67848964098d703fb0c16c75b8679219ef3960ca7d55d0563b168ee8c361a8d"


class LinearFixture:
    """Algebraic interface fixture, never a neural model or physical dynamics."""
    def __init__(self, task):
        self.task = task
        self.j = np.zeros((192, 10))
        self.j[0, 0], self.j[1, 1] = 1., 2.
        self.calls = 0

    def step(self, context, actions):
        self.calls += 1
        c = deepcopy(context)
        raw = np.asarray(actions, dtype=np.float32).reshape(5, 2)
        latent = np.asarray(c["latent_history"][-1]) + self.j @ raw.reshape(10)
        token = core.normalized_tokens(self.task, raw)[0]
        c["latent_history"] = c["latent_history"][1:] + [latent.tolist()]
        c["action_history"] = c["action_history"][1:] + [token.tolist()]
        return c

    def replay(self, context, actions):
        for packet in np.asarray(actions).reshape(-1, 5, 2):
            context = self.step(context, packet)
        return context

    def nominal_jacobian(self, context, operation):
        nominal = self.step(context, np.zeros((5, 2), np.float32))
        operation("jacobian", lambda: self.step(context, np.zeros((5, 2), np.float32)))
        return np.asarray(nominal["latent_history"][-1]), self.j.copy()


def audit():
    saved = run.verify_saved(PREPARED, PREPARE_SHA, RESULT)
    core.require(saved["report_sha256"] == REPORT_SHA, "frozen report identity")
    exported = run.b.read(RESULT / "task.json.gz")
    interfaces = run.b.load_task(RESULT / "task.json.gz", TASK_SHA)
    task = interfaces["gn_task"]
    core.require(core.task_id(task) == exported["task_id"], "task identity")
    core.require(exported["development_seed"] == 301000001 and
                 exported["new_independent_confirmation"] is False and
                 exported["P1_admission"] is False, "development only")
    raw = np.asarray(task["root_raw_action_history"], np.float32)
    tokens = core.normalized_tokens(task, raw)
    core.require(np.array_equal(tokens, task["root_context"]["action_history"]), "real raw normalization")
    core.require(np.array_equal(np.asarray(task["projection"]), np.eye(192)), "identity Q")
    config = dict(method="full_H2_GN", B1=1, B2=2,
                  initial_logits=[np.zeros((2, 10), np.float32).tolist()],
                  step_scale=1., damping=1., extra_line_search_trials=0)
    screen.validate_configuration(config, task)
    packet = interfaces["d2p_packet"]
    q = np.asarray(interfaces["q"])
    core.require(packet["root_context"] == task["root_context"] and packet["goal"] == task["goal"],
                 "planner receives exact root and goal")
    fixture = LinearFixture(task)
    result = compiler.compile_arm(packet, q, fixture, lambda _: None)
    winner = compiler.validate_compilation(packet, q, result)
    core.require(fixture.calls == 66 and len(result["candidates"]) == 10
                 and result["whole_and_segmented_exact"], "fixture compiler path")
    return dict(schema="p1_queue_goal_task_audit_20261101_codex.v1",
                status="h2_interface_and_algebraic_fixture_verified",
                report_sha256=REPORT_SHA, task_sha256=TASK_SHA, task_id=exported["task_id"],
                seed=exported["development_seed"], context_shape=[3, 192],
                action_history_shape=[2, 10], raw_actions_shape=[10, 2],
                goal_shape=[192], projection_shape=[192, 192], horizon=2, packet_steps=5,
                normalizer=exported["task_contract"]["normalization"], actual_rgb_steps=[0, 5, 10],
                goal_from_same_instance_reset=True,
                fixture=dict(backend="explicit non-neural linear algebra stand-in",
                             candidates=len(result["candidates"]), backend_step_calls=fixture.calls,
                             whole_segmented_replay_exact=result["whole_and_segmented_exact"],
                             winner_index=winner["index"], scientific_interpretation="none"),
                compatibility="H2 task load and compiler packet interface only; not H4 physical runner readiness",
                with_target_semantics="false excludes privileged target state; goal RGB/latent is present",
                historical_coverage="unknown", training_membership="unknown",
                independent_confirmation=False, P1_admission=False,
                new_cuda_calls=0, neural_calls=0, physics_calls=0, training=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    core.require(not output.exists(), "exclusive audit output")
    result = audit()
    result["auditor_sha256"] = run.b.sha(Path(__file__))
    with output.open("xb") as f:
        f.write(run.b.canonical(result))
        f.flush()
        os.fsync(f.fileno())
    print(run.b.canonical(dict(path=str(output), sha256=run.b.sha(output), **result)).decode().strip())


if __name__ == "__main__":
    main()
