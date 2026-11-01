import numpy as np
import pytest

from experiments.spectral_reachability import agent_p1_projection_geometry_20261029_v1 as m


def panel():
    x = np.zeros(10)
    x[0] = m.refinement.F.CAP
    candidate = {"logits": x.tolist(), "raw_bits": m.refinement.F.bits(m.refinement.F.chart(x)),
                 "terminal_Q": 2.0, "uphill_from_initial": True,
                 "better_terminal_than_original": False, "witness": False}
    gradient = {"raw_gradient": [[-1.0, 0.0]] + [[0.0, 0.0]] * 4}
    result = {"initial_Q": 1.0, "stage_Q": 2.0, "baseline_Q": 1.0,
              "iterates": [candidate.copy() for _ in range(9)],
              "gradients": [gradient.copy() for _ in range(8)]}
    return {"rows": [{"result": result}]}, {"group_denominator": 1, "candidate_denominator": 9}


def test_outward_direction_is_not_inward():
    report, receipt = panel()
    result = m.analyze(report, receipt)
    assert result["steps"] == result["near_boundary_outward_descent"] == 8
    assert result["tangent_ratio_max"] == pytest.approx(0)
    assert result["linear_comparison_count"] == 8


def test_tangent_comparison_is_only_a_linear_prediction():
    report, receipt = panel()
    for gradient in report["rows"][0]["result"]["gradients"]:
        gradient["raw_gradient"] = [[-1.0, 1.0]] + [[0.0, 0.0]] * 4
    result = m.analyze(report, receipt)
    assert result["tangent_lower_linear_delta"] == 8
    assert result["P1_admission"] is result["physical_release"] is False


def test_rejects_corrupted_action_bits():
    report, receipt = panel()
    report["rows"][0]["result"]["iterates"][0]["raw_bits"] = "bad"
    with pytest.raises(ValueError, match="identity"):
        m.analyze(report, receipt)


def test_rejects_invalid_tolerance():
    report, receipt = panel()
    with pytest.raises(ValueError, match="tolerance"):
        m.analyze(report, receipt, float("nan"))
