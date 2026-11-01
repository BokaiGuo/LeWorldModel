import math

import pytest

from experiments.spectral_reachability.p1_matched_censored_analysis_20261030_main import errors, paired


def test_errors_keep_position_and_wrapped_angle_separate():
    got = errors([0, 0, 3, 4, math.pi - 0.1, 0, 0], [0, 0, 0, 0, -math.pi + 0.1, 0, 0])
    assert got["position"] == 5
    assert got["angle"] == pytest.approx(0.2)
    assert errors(None, [0] * 7) is None


def test_fixed_pairs_preserve_censor_and_reject_mismatched_actions():
    anchors, probes, rows = [], [], []
    for i in range(16):
        anchors.append(dict(index=i, root_slot=i // 2, condition="on", arm="x", task_id=str(i // 2),
                            goal=[0] * 7, physical_root_sha256=str(i // 2),
                            prefix5_float32_hex=str(i % 2),
                            role=("incumbent" if i % 2 == 0 else "model_selected")))
        probes.append([dict(index=j, name=str(j), raw5_float32_hex=str(j)) for j in range(21)])
        rows.append(dict(anchor=i, statuses=["probe_completed"] * 21,
                         endpoints=[[0, 0, 1 + i % 2, 0, 0, 0, 0] for _ in range(21)]))
    rows[1]["statuses"][2] = "censored_out_of_domain"
    rows[1]["endpoints"][2] = None
    result = paired(anchors, probes, rows)
    assert result[0]["probes"][0]["selected_minus_incumbent"]["position"] == 1
    assert result[0]["probes"][0]["selected_minus_incumbent_gain_vs_own_zero"]["position"] == 0
    assert result[0]["probes"][2]["selected_minus_incumbent"] is None
    assert result[0]["probes"][2]["selected_minus_incumbent_gain_vs_own_zero"] is None
    assert result[0]["duplicate_group"] != result[1]["duplicate_group"]
    for i in (2, 3):
        anchors[i]["task_id"] = anchors[i - 2]["task_id"]
        anchors[i]["physical_root_sha256"] = anchors[i - 2]["physical_root_sha256"]
    assert paired(anchors, probes, rows)[0]["duplicate_group"] == paired(anchors, probes, rows)[1]["duplicate_group"]
    rows[1]["endpoints"][0] = None
    rows[1]["statuses"][0] = "censored_out_of_domain"
    assert paired(anchors, probes, rows)[0]["probes"][1]["selected_minus_incumbent_gain_vs_own_zero"] is None
    rows[1]["statuses"][1] = "censored_out_of_domain"
    with pytest.raises(ValueError, match="status endpoint mismatch"):
        paired(anchors, probes, rows)
    rows[1]["statuses"][1] = "probe_completed"
    probes[1][0]["raw5_float32_hex"] = "different"
    with pytest.raises(ValueError, match="matched probe"):
        paired(anchors, probes, rows)
