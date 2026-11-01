"""Array-only regression: no model inference or physical execution."""
import numpy as np
import pytest

from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as p1
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
from experiments.spectral_reachability import pusht_task_spectral_bank as bank
from experiments.spectral_reachability import agent_p2_wallcapped_cuda_20261029_v1 as p2


class ArrayMeter:
    def call(self, _name, fn):
        return fn()

    def emit(self, *_args, **_kwargs):
        pass


def fixture(rank_deficient):
    rng = np.random.default_rng(2718)
    j = rng.normal(size=(192, 10))
    j[:, 0] *= -1
    if rank_deficient:
        j[:, -1] = j[:, 0] + j[:, 1]
    target = rng.normal(size=192)
    return j, target


@pytest.mark.parametrize("rank_deficient", [False, True])
@pytest.mark.parametrize("cap", [0.05, 10.])
def test_p1_compiler_and_p2_ridge_match_normal_equations(rank_deficient, cap):
    j, target = fixture(rank_deficient)
    task = {"projection": np.eye(192).tolist(), "goal": target.tolist(),
            "packet_logit_cap": cap}
    result = p1.compile_response(task, j, np.zeros(192), cap, ArrayMeter())
    a, r = j / np.sqrt(192), target / np.sqrt(192)
    multiplier = result["ball_multiplier"]
    expected = np.linalg.solve(a.T @ a + (p1.PROTOCOL["damping"] + multiplier) * np.eye(10),
                               a.T @ r)
    # Production deliberately rounds the step to float32 and moves boundary
    # points inward by up to four float32 epsilons.
    np.testing.assert_allclose(result["step"], expected, rtol=1e-6, atol=2e-8)
    assert np.linalg.norm(result["step"]) <= cap + 1e-10
    assert (multiplier > 0) == (cap == 0.05)

    # The P2 helper uses its own fixed cap and unscaled Jacobian.
    left, singular, right = np.linalg.svd(j, full_matrices=False)
    p2_target = target * (100. if cap == .05 else 1.)
    p2_multiplier, p2_step = p2.ridge_from_svd(left, singular, right, p2_target)
    direct = np.linalg.solve(j.T @ j +
                             (compiler.DAMPING + p2_multiplier) * np.eye(10),
                             j.T @ p2_target)
    np.testing.assert_allclose(p2_step, direct, rtol=1e-9, atol=1e-9)
    assert np.linalg.norm(p2_step) <= p2.old.CAP_LOGIT + 1e-10
    assert (p2_multiplier > 0) == (cap == .05)


def test_paired_canonicalization_matches_damped_equation():
    rng = np.random.default_rng(42)
    j = rng.normal(size=(192, 2))
    j[:, 0] *= -1
    goal = rng.normal(size=192) * .01
    result = bank.synthesize(jacobian=j, nominal_terminal=np.zeros(192),
                             goal=goal, projection=np.eye(192))
    expected = np.linalg.solve(j.T @ j + .01 * np.eye(2), j.T @ goal)
    assert np.linalg.norm(expected) < result["progress_norm_cap"]
    np.testing.assert_allclose(np.asarray(result["params"])[1, 0], expected, atol=1e-12)
