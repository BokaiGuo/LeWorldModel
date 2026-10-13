"""Audit the decomposed finite-horizon transfer-risk theorem.

This runner produces only synthetic theorem evidence: a tightness witness, an
explicit spectrum/alignment/commutator counterexample, a non-normal transient
comparison, and a local nonlinear remainder check.  It must not be reported as
learned-model or planning evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from finite_horizon_transfer import (
    adversarial_matched_transfer_counterexample,
    finite_horizon_transfer_risk_decomposition,
    spectrum_alignment_commutator_counterexample,
    transfer_bound_tightness_example,
)
from nonlinear_transfer_certificate import (
    adversarial_nonlinear_remainder_counterexample,
    check_nonlinear_transfer_applicability,
)


def nonlinear_local_remainder_audit() -> dict[str, float | bool]:
    """Check a second-order local Taylor remainder for a bounded tube."""

    A = np.asarray([[0.75, 0.10], [0.0, 0.65]], dtype=np.float64)
    beta = 0.08
    x = np.asarray([0.12, -0.07], dtype=np.float64)
    delta = np.asarray([0.004, -0.003], dtype=np.float64)

    def function(value: np.ndarray) -> np.ndarray:
        return A @ value + beta * np.sin(value)

    jacobian = A + beta * np.diag(np.cos(x))
    remainder = function(x + delta) - function(x) - jacobian @ delta
    # For beta*sin(.) the Euclidean Hessian operator norm is at most beta.
    bound = 0.5 * beta * float(np.linalg.norm(delta)) ** 2
    return {
        "remainder_norm": float(np.linalg.norm(remainder)),
        "certified_local_bound": float(bound),
        "bound_holds": bool(np.linalg.norm(remainder) <= bound + 1.0e-12),
        "tube_radius": float(np.linalg.norm(delta)),
        "applicability_condition": "F is C2 and the trajectory remains in the declared local tube",
    }


def run() -> dict[str, object]:
    normal = np.diag([0.9, 0.4]).astype(np.float64)
    nonnormal = np.asarray([[0.9, 1.0], [0.0, 0.9]], dtype=np.float64)
    D = np.eye(2, dtype=np.float64)
    h = 6
    residuals = np.full(h, 0.05, dtype=np.float64)
    normal_bound = finite_horizon_transfer_risk_decomposition(
        A=normal, D=D, horizon=h, closure_residuals=residuals,
        off_manifold_residuals=np.zeros(h), action_effect_errors=np.zeros(h),
        readout_error=0.0, readout_state_bound=0.0,
    )
    nonnormal_bound = finite_horizon_transfer_risk_decomposition(
        A=nonnormal, D=D, horizon=h, closure_residuals=residuals,
        off_manifold_residuals=np.zeros(h), action_effect_errors=np.zeros(h),
        readout_error=0.0, readout_state_bound=0.0,
    )
    nonlinear_certificate = check_nonlinear_transfer_applicability(
        local_jacobians=[np.asarray([[0.8]], dtype=np.float64)] * 3,
        task_readout_jacobian=np.asarray([[1.0]], dtype=np.float64),
        jacobian_lipschitz=[0.1, 0.1, 0.1],
        tube_radius=0.1,
        residual_envelopes={
            "closure": [0.01, 0.01, 0.01],
            "off_manifold": [0.01, 0.01, 0.01],
            "action": [0.01, 0.01, 0.01],
        },
        action_support={"passed": True, "coverage": 1.0},
        independent_transition_consistency={"passed": True, "split": "planner_reachable"},
        independent_readout_calibration={"passed": True, "split": "readout_holdout"},
        remainder_budget=0.01,
    )
    return {
        "status": "synthetic_theory_audit_only",
        "tightness": transfer_bound_tightness_example(),
        "counterexample": spectrum_alignment_commutator_counterexample(),
        "adversarial_matched_counterexample": adversarial_matched_transfer_counterexample(),
        "normal_vs_nonnormal": {
            "same_spectral_radius": True,
            "normal": normal_bound,
            "nonnormal": nonnormal_bound,
            "nonnormal_transient_excess_larger": bool(
                nonnormal_bound["components"]["E_transient"]
                > normal_bound["components"]["E_transient"]
            ),
        },
        "nonlinear_local_remainder": nonlinear_local_remainder_audit(),
        "learned_nonlinear_applicability": nonlinear_certificate,
        "adversarial_nonlinear_counterexample": adversarial_nonlinear_remainder_counterexample(),
        "claim_boundary": (
            "These checks validate the algebraic bound, its tight scalar case, "
            "and its counterexamples. They do not establish learned-model "
            "transfer-risk prediction or planning benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "finite_horizon_transfer_risk_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
