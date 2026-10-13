import numpy as np

from experiments.theory_spectral_transfer.risk_aware_control import (
    certified_metric_tube,
    covariant_transfer_risk_mpc,
    markov_parameters,
)


def _errors(horizon: int, value: float = 0.01) -> dict[str, object]:
    return {
        "closure_residuals": [value] * horizon,
        "off_manifold_residuals": [value] * horizon,
        "action_effect_errors": [value] * horizon,
        "readout_error": value,
        "readout_state_bound": 1.0,
    }


def test_markov_parameters_match_declared_g_h():
    A = np.asarray([[0.8]])
    B = np.asarray([[2.0]])
    D = np.asarray([[3.0]])
    values = markov_parameters(A, B, D, 3)
    assert np.allclose(values[0], [[6.0]])
    assert np.allclose(values[2], [[3.0 * 0.8**2 * 2.0]])


def test_controller_permits_only_after_both_independent_gates():
    A = np.asarray([[0.5]])
    B = np.asarray([[1.0]])
    D = np.asarray([[1.0]])
    candidates = [np.zeros((2, 1)), np.ones((2, 1))]
    kwargs = dict(
        A=A, B=B, D=D, metric=np.eye(1), x0=np.zeros(1), goal_task=np.asarray([1.0]),
        candidate_actions=candidates, risk_errors=[_errors(2), _errors(2)],
        risk_threshold=10.0, risk_weight=0.1,
        action_identifiability={"passed": True}, transition_consistency={"passed": True},
        tube_certificate={"status": "certified"},
    )
    permitted = covariant_transfer_risk_mpc(**kwargs)
    assert permitted["status"] == "permitted"
    assert permitted["gates"] == {
        "action_identifiability": True,
        "transition_consistency": True,
        "local_metric_tube": True,
    }
    blocked = covariant_transfer_risk_mpc(**{**kwargs, "transition_consistency": {"passed": False}})
    assert blocked["status"] == "fallback_fail_closed"
    assert blocked["selected_index"] == 0


def test_controller_fails_closed_when_independent_risk_envelope_is_missing():
    report = covariant_transfer_risk_mpc(
        A=np.asarray([[0.5]]), B=np.asarray([[1.0]]), D=np.asarray([[1.0]]), metric=np.eye(1),
        x0=np.zeros(1), goal_task=np.asarray([1.0]), candidate_actions=[np.zeros((2, 1))],
        risk_errors=[None], risk_threshold=10.0, risk_weight=0.0,
        action_identifiability={"passed": True}, transition_consistency={"passed": True},
        tube_certificate={"status": "certified"},
    )
    assert report["status"] == "fallback_fail_closed"
    assert report["candidate_reports"][0]["risk_bound"] is None


def test_metric_tube_certificate_is_rejected_outside_local_region():
    certified = certified_metric_tube(
        A=np.asarray([[0.5]]), B=np.asarray([[1.0]]), metric=np.eye(1),
        action_bound=0.1, disturbance_bound=0.1, local_tube_radius=1.0,
    )
    assert certified["status"] == "certified"
    rejected = certified_metric_tube(
        A=np.asarray([[0.5]]), B=np.asarray([[1.0]]), metric=np.eye(1),
        action_bound=1.0, disturbance_bound=1.0, local_tube_radius=1.0,
    )
    assert rejected["status"] == "rejected_outside_local_tube"
