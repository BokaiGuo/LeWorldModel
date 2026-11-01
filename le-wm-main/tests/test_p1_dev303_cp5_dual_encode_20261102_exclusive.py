import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_cp5_dual_encode_20261102_exclusive as subject,
)


def test_cpu_contract_has_two_arms_no_planning_or_physics():
    value = subject.protocol()
    assert [row["role"] for row in value["roles"]] == list(subject.parent.ARMS)
    assert value["rgb_encode_calls"] == 2
    assert value["planning"] is value["physical_execution"] is False
    assert value["P1_admission"] is value["independent_confirmation"] is False


def test_context_uses_actual_prefix_and_sealed_warmup_history():
    value = subject.protocol()
    task = value["task"]
    for role in value["roles"]:
        latent = np.arange(task["latent_dim"], dtype=np.float32)
        states, actions = subject.contexts(task, role["actual_prefix"], latent)
        assert np.array_equal(states[-1], latent)
        assert np.allclose(states[:2], np.asarray(task["root_context"]["latent_history"][-2:]))
        assert np.allclose(actions[0], task["root_context"]["action_history"][-1])
        expected = ((np.asarray(role["actual_prefix"], np.float32)
                     - np.asarray(task["action_mean"], np.float32))
                    / np.asarray(task["action_std"], np.float32)).reshape(10)
        assert np.array_equal(actions[1], expected)


def test_context_rejects_bad_latent():
    value = subject.protocol()
    with pytest.raises(ValueError, match="actual prefix/cp5 latent"):
        subject.contexts(value["task"], value["roles"][0]["actual_prefix"], [float("nan")])
