import numpy as np
import pytest

from experiments.spectral_reachability.p1_task_response_saved_panel_20261030_main import analyze


def fixtures():
    task = dict(projection=np.eye(192).tolist(), goal=[1.] + [0.] * 191,
                root_context=dict(latent_history=[[0.] * 192]))
    jac = np.zeros((192, 10))
    candidate = dict(stage_Q=1., signal=dict(jacobian=jac.tolist()))
    arms = [dict(slot=i // 2, arm=("projected_ascent" if i % 2 == 0 else "outward_free_tangent"),
                 result=dict(candidates=[candidate] * 20)) for i in range(8)]
    return dict(rows=arms), dict(requests=[dict(task=task)] * 4)


def test_full_fixed_panel_and_stable_ties():
    report, prep = fixtures()
    result = analyze(report, prep)
    assert len(result["arms"]) == 8
    assert result["arms"][0]["selected_indices"] == [0, 1, 2, 3]
    assert result["P1_admission"] is False


def test_refuses_metric_change():
    report, prep = fixtures()
    prep["requests"][0]["task"]["projection"][0][0] = 2.
    with pytest.raises(ValueError, match="identity target metric"):
        analyze(report, prep)


def test_refuses_duplicate_arm_in_a_slot():
    report, prep = fixtures()
    report["rows"][1]["arm"] = report["rows"][0]["arm"]
    with pytest.raises(ValueError, match="fixed panel identities"):
        analyze(report, prep)
