import pytest

from experiments.pc_state_effect_precision_audit_20261106_codex import reconstruct


def test_quadratic_reconstruct_zero_error():
    f = lambda x: 2 + 3*x + 4*x*x
    steps = (.003, .03, .1)
    scan = {"center": f(0.), "scan": [
        {"step": h, "minus": f(-h), "plus": f(h),
         "central_derivative": (f(h) - f(-h))/(2*h),
         "curvature": (f(h) + f(-h) - 2*f(0.))/h**2}
        for h in steps]}
    assert reconstruct(scan) == pytest.approx(0., abs=1e-11)


def test_reject_corrupted_curvature():
    scan = {"center": 1., "scan": [{"step": .003, "minus": 1., "plus": 1.,
                                    "central_derivative": 0., "curvature": 4.}]}
    with pytest.raises(AssertionError):
        reconstruct(scan)
