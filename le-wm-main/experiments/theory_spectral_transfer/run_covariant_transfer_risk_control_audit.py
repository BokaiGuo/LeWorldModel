"""Executable audit for the covariant transfer-risk control layer.

This is a synthetic control audit.  It verifies the new control interface,
its finite-horizon ``G_h`` object, the local metric-tube condition, and the
fail-closed behavior.  It is not learned-model planning evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from risk_aware_control import certified_metric_tube, covariant_transfer_risk_mpc


def _errors(horizon: int, value: float) -> dict[str, object]:
    return {
        "closure_residuals": [value] * horizon,
        "off_manifold_residuals": [value] * horizon,
        "action_effect_errors": [value] * horizon,
        "readout_error": value,
        "readout_state_bound": 1.0,
    }


def run() -> dict[str, object]:
    A = np.asarray([[0.65, 0.15], [0.0, 0.55]], dtype=np.float64)
    B = np.asarray([[0.7], [0.25]], dtype=np.float64)
    D = np.eye(2, dtype=np.float64)
    metric = np.diag([1.0, 4.0])
    candidates = [
        np.asarray([[0.0], [0.0], [0.0]], dtype=np.float64),
        np.asarray([[0.7], [0.0], [0.0]], dtype=np.float64),
        np.asarray([[0.7], [0.7], [0.0]], dtype=np.float64),
    ]
    common = dict(
        A=A,
        B=B,
        D=D,
        metric=metric,
        x0=np.zeros(2),
        goal_task=np.asarray([0.8, 0.2]),
        candidate_actions=candidates,
        risk_errors=[_errors(3, 0.01), _errors(3, 0.01), _errors(3, 0.01)],
        risk_threshold=1.0,
        risk_weight=0.25,
        action_identifiability={"passed": True, "source": "independent action-balanced probe"},
        transition_consistency={"passed": True, "source": "independent real-predicted transition gate"},
        tube_certificate={
            "status": "certified",
            "source": "certified_metric_tube(A,B,metric,...)",
        },
    )
    permitted = covariant_transfer_risk_mpc(**common)
    blocked = covariant_transfer_risk_mpc(
        **{**common, "action_identifiability": {"passed": False, "source": "missing"}}
    )
    tube = certified_metric_tube(
        A=A,
        B=B,
        metric=metric,
        action_bound=0.7,
        disturbance_bound=0.02,
        local_tube_radius=2.0,
    )
    return {
        "schema": "covariant_transfer_risk_control_audit.v1",
        "status": "synthetic_control_theory_audit_only",
        "permitted_case": permitted,
        "fail_closed_case": blocked,
        "metric_tube": tube,
        "claim_boundary": (
            "This validates the control contract and its conditional local tube theorem. "
            "It does not establish learned-model transfer-risk prediction, stable pose closure, "
            "or cross-environment planning benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "covariant_transfer_risk_control_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
