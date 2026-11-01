from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_fixed_suffix_rank_analysis_20261030 as a


def data():
    models, rows = [], []
    for i in range(8):
        models.append(dict(slot=i, arm="a", indices=[0, 1, 2, 3],
                           scores=[0., 1., 2., 3.], selected_index=0))
        for j in range(4):
            rows.append(dict(arm="a", role=f"candidate_{j}",
                             plan=dict(slot=i, candidate_index=j),
                             status="horizon_completed",
                             responses=[dict(step=15, native_success=False,
                                             native_metrics=dict(object_position_error=float(j),
                                                                 wrapped_object_angle_error=float(3-j)))]))
    return {"rows": rows}, {"rows": models}


def test_rank_and_censor():
    physical, model = data()
    physical["rows"][1]["status"] = "censored_out_of_domain"
    result = a.analyze(physical, model)["groups"][0]
    assert result["candidates"][1]["endpoint"] is None
    assert result["comparisons"]["object_position_error"]["concordant"] == 3
    assert result["comparisons"]["wrapped_object_angle_error"]["discordant"] == 3


def test_wrong_binding_is_rejected():
    physical, model = data()
    altered = deepcopy(physical)
    altered["rows"][0]["plan"]["candidate_index"] = 3
    with pytest.raises(Exception):
        a.analyze(altered, model)
