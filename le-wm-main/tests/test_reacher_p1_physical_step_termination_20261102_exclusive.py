"""CPU-only reproduction of the inherited physical executor's accounting gap."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_executor():
    source = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_execute")
    module = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "restore_execution_snapshot": lambda env, seed, snapshot: ("root", "goal"),
        "_verify_execution_identity": lambda *args: {"verified": "fixture"},
        "_qpos_distance": lambda a, b: float(np.linalg.norm(a - b)),
    }
    code = compile(module, str(source), "exec", flags=__import__("__future__").annotations.compiler_flag)
    exec(code, namespace)
    return namespace["_execute"]


class FakeEnv:
    def __init__(self, *, stop_at=None, truncated=False, fail_at=None):
        self.steps = 0
        self.stop_at = stop_at
        self.truncated = truncated
        self.fail_at = fail_at
        self.unwrapped = SimpleNamespace(env=SimpleNamespace(
            physics=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2))),
            task=SimpleNamespace(target_qpos=np.ones(2)),
        ))

    def step(self, action):
        self.steps += 1
        if self.steps == self.fail_at:
            raise RuntimeError("fixture step error")
        stop = self.steps == self.stop_at
        return None, 0.0, stop and not self.truncated, stop and self.truncated, {}


def execute(env, receipt=None):
    return load_executor()(env, 1, np.zeros((3, 2), dtype=np.float32),
        expected_frame="root", expected_goal="goal", execution_snapshot={},
        execution_receipt=receipt)


def test_complete_preserves_old_return_shape_and_receipt():
    env, receipt = FakeEnv(), {}
    physical, identity = execute(env, receipt)
    assert env.steps == 3
    assert set(physical) == {"initial_qpos_distance", "final_qpos_distance",
                             "min_qpos_distance", "physical_progress"}
    assert identity == {"verified": "fixture"}
    assert receipt == {"requested_steps": 3, "attempted_steps": 3, "returned_steps": 3,
                       "terminated": False, "truncated": False, "status": "completed"}
    assert execute(FakeEnv()) == (physical, identity)


@pytest.mark.parametrize("truncated", [False, True])
def test_early_termination_fails_closed(truncated):
    env, receipt = FakeEnv(stop_at=1, truncated=truncated), {}
    with pytest.raises(RuntimeError, match="returned=1 requested=3"):
        execute(env, receipt)
    assert env.steps == 1
    assert receipt == {"requested_steps": 3, "attempted_steps": 1, "returned_steps": 1,
                       "terminated": not truncated, "truncated": truncated, "status": "early_termination"}


@pytest.mark.parametrize("truncated", [False, True])
def test_terminal_on_last_step_is_complete_and_recorded(truncated):
    env, receipt = FakeEnv(stop_at=3, truncated=truncated), {}
    execute(env, receipt)
    assert receipt["status"] == "completed"
    assert receipt["returned_steps"] == 3
    assert receipt["terminated"] is (not truncated)
    assert receipt["truncated"] is truncated


def test_failed_step_debited_without_return():
    env, receipt = FakeEnv(fail_at=2), {}
    with pytest.raises(RuntimeError, match="fixture step error"):
        execute(env, receipt)
    assert receipt["attempted_steps"] == 2
    assert receipt["returned_steps"] == 1
    assert receipt["status"] == "step_error"


def test_rejects_empty_or_nonfinite_actions_and_nonempty_receipt():
    execute_fn = load_executor()
    env = FakeEnv()
    for actions in (np.zeros((0, 2)), np.array([[float("nan"), 0.0]])):
        with pytest.raises(ValueError, match="nonempty finite"):
            execute_fn(env, 1, actions, expected_frame="root", expected_goal="goal", execution_snapshot={})
    with pytest.raises(ValueError, match="empty dict"):
        execute(env, {"old": 1})
    assert env.steps == 0


def test_real_module_executor_uses_the_same_fail_closed_path(monkeypatch):
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
    monkeypatch.setattr(p0, "restore_execution_snapshot", lambda env, seed, snapshot: ("root", "goal"))
    monkeypatch.setattr(p0, "_verify_execution_identity", lambda *args: {"verified": "fixture"})
    env, receipt = FakeEnv(stop_at=2), {}
    with pytest.raises(RuntimeError, match="returned=2 requested=3"):
        p0._execute(env, 1, np.zeros((3, 2)), expected_frame="root", expected_goal="goal",
                    execution_snapshot={}, execution_receipt=receipt)
    assert receipt["attempted_steps"] == receipt["returned_steps"] == 2
    assert receipt["terminated"] and receipt["status"] == "early_termination"
