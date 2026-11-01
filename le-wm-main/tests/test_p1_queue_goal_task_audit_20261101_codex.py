from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_queue_goal_task_audit_20261101_codex as a


def test_actual_export_to_h2_compiler_fixture():
    result = a.audit()
    assert result["status"] == "h2_interface_and_algebraic_fixture_verified"
    assert result["fixture"]["candidates"] == 10
    assert result["fixture"]["backend_step_calls"] == 66
    assert result["neural_calls"] == result["physics_calls"] == 0
    assert result["independent_confirmation"] is False


def test_loader_and_action_tamper_fail_closed():
    with pytest.raises(ValueError):
        a.run.b.load_task(a.RESULT / "task.json.gz", "0" * 64)
    task = a.run.b.load_task(a.RESULT / "task.json.gz", a.TASK_SHA)["gn_task"]
    changed = deepcopy(task)
    changed["root_context"]["action_history"][0][0] += 0.1
    with pytest.raises(ValueError):
        a.core.validate_task(changed)
    changed = deepcopy(task)
    changed["horizon"] = 4
    with pytest.raises(ValueError):
        a.core.validate_task(changed)
    raw = np.asarray(task["root_raw_action_history"], np.float32)
    np.testing.assert_array_equal(a.core.normalized_tokens(task, raw),
                                  task["root_context"]["action_history"])
