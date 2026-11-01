from copy import deepcopy

import pytest

from experiments.spectral_reachability.p1_null_same_action_goal_join_20261030_main import join


def fixture():
    new, phys, old, oldp = ({"rows": []} for _ in range(4))
    for i in range(16):
        role = "selected_prefix" if i % 2 == 0 else "zero_prefix"
        vec = [float(i % 2)] + [0.] * 191
        comps = [dict(step=t, actual_latent=vec, open_prediction=vec,
                      refreshed_prediction=None if t == 5 else vec) for t in (5, 10, 15)]
        new["rows"].append(dict(index=i, task_id=str(i // 2), slot=i // 2, arm="a",
                                role=role, duplicate_action_task_group=i // 2, comparisons=comps))
        response = dict(step=0, native_metrics=dict(object_position_error=1.,
                        wrapped_object_angle_error=.1), native_success=False)
        responses = [dict(response, step=j) for j in range(1, 16)]
        phys["rows"].append(dict(index=i, task_id=str(i // 2), role=role,
                                 goal_pixel_sha256="goal", responses=responses))
        old["rows"].append(dict(slot=i // 2, arm="a", role="model_selected", goal_latent=[0.] * 192))
        oldp["rows"].append(dict(slot=i // 2, arm="a", role="model_selected",
                                 task_id=str(i // 2), goal_pixel_sha256="goal"))
    return new, phys, old, oldp


def test_paired_goal_difference_and_refresh_missing():
    result = join(*fixture())
    times = result["pairs"][0]["times"]
    assert times[0]["selected_minus_zero"]["actual_latent_goal_distance"] == -1.
    assert times[0]["selected_minus_zero"]["refreshed_prediction_goal_distance"] is None
    assert times[1]["selected_minus_zero"]["refreshed_prediction_goal_distance"] == -1.


def test_reject_goal_mismatch_and_time_mismatch():
    parts = fixture()
    broken = deepcopy(parts)
    broken[1]["rows"][0]["goal_pixel_sha256"] = "wrong"
    with pytest.raises(ValueError, match="goal join"):
        join(*broken)
    broken = deepcopy(parts)
    broken[1]["rows"][0]["responses"][4]["step"] = 4
    with pytest.raises(ValueError, match="physical time"):
        join(*broken)
