import numpy as np
import pytest

from experiments.spectral_reachability import p1_carry_binding_20261101 as binding


def test_identity_and_cost():
    plan = np.arange(30, dtype=np.float32).reshape(3, 5, 2) / 30
    sealed = binding.BoundPlan.seal("root0", 5, plan)
    result = sealed.bind_feedback(episode="root0", boundary=10,
                                  issued_packet=plan[0], rgb_sha256="a" * 64)
    np.testing.assert_array_equal(result["tail"], plan[1:])
    assert binding.extra_cost(8, 2, has_carry=True) == {
        "action_encode_calls": 18, "predict_calls": 18, "predict_samples": 18,
        "replay_calls": 9, "selection_evaluations": 9, "backward_calls": 0}


@pytest.mark.parametrize("change", ["episode", "boundary", "packet", "rgb"])
def test_invalid_feedback_is_rejected(change):
    plan = np.zeros((3, 5, 2), dtype=np.float32)
    sealed = binding.BoundPlan.seal("root0", 5, plan)
    args = dict(episode="root0", boundary=10, issued_packet=plan[0], rgb_sha256="a" * 64)
    if change == "episode":
        args["episode"] = "root1"
    elif change == "boundary":
        args["boundary"] = 15
    elif change == "packet":
        args["issued_packet"] = np.ones((5, 2), dtype=np.float32)
    else:
        args["rgb_sha256"] = "bad"
    with pytest.raises(ValueError):
        sealed.bind_feedback(**args)


def test_mutation_and_wrong_dtype_are_rejected():
    plan = np.zeros((2, 5, 2), dtype=np.float32)
    sealed = binding.BoundPlan.seal("root0", 0, plan)
    plan[0] = 1  # Caller cannot mutate the sealed copy.
    with pytest.raises(ValueError):
        sealed.bind_feedback(episode="root0", boundary=5, issued_packet=plan[0],
                             rgb_sha256="a" * 64)
    with pytest.raises(ValueError):
        binding.BoundPlan.seal("root0", 0, plan.astype(np.float64))


def test_exhausted_tail_is_explicit():
    plan = np.zeros((1, 5, 2), dtype=np.float32)
    result = binding.BoundPlan.seal("r", 0, plan).bind_feedback(
        episode="r", boundary=5, issued_packet=plan[0], rgb_sha256="b" * 64)
    assert result["tail"].shape == (0, 5, 2)
