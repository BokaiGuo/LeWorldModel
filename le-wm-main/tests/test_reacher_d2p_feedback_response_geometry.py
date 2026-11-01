import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import audit_reacher_d2p_feedback_response_geometry as audit


def endpoints(scale):
    result = []
    for i, plus, minus in audit.probes(scale):
        result.append({"coordinate": i, **{name: {"model": np.full(192, value[i]).tolist(),
            "observed": np.full(192, 2 * value[i]).tolist(), "qpos": [float(value[i]), 0.],
            "raw_actions": value.reshape(10, 2).tolist()} for name, value in (("plus", plus), ("minus", minus))}})
    return result


@pytest.mark.parametrize("scale", audit.SCALES)
def test_all_twenty_both_signs(scale):
    p = audit.probes(scale)
    assert len(p) == 20
    assert [i for i, _, _ in p] == list(range(20))
    for i, plus, minus in p:
        assert np.count_nonzero(plus) == 1 and plus[i] == np.float32(scale)
        np.testing.assert_array_equal(plus, -minus)


def test_arbitrary_probe_scale_rejected():
    with pytest.raises(ValueError):
        audit.probes(.2)


@pytest.mark.parametrize("scale", audit.SCALES)
def test_finite_difference_complete_coordinates(scale):
    matrices = audit.validate_endpoints(endpoints(scale), scale)
    assert matrices["model_fd"].shape == (192, 20)
    np.testing.assert_allclose(matrices["model_fd"], 1)
    np.testing.assert_allclose(matrices["physical_rgb_fd"], 2)
    np.testing.assert_allclose(matrices["physical_qpos_fd"][0], 1)
    np.testing.assert_allclose(matrices["physical_qpos_fd"][1], 0)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "projection", "nan", "outside"])
def test_invalid_probe_rejected(mutation):
    data = copy.deepcopy(endpoints(.1))
    if mutation == "missing":
        data.pop()
    elif mutation == "duplicate":
        data[-1]["coordinate"] = 0
    elif mutation == "projection":
        data[0]["plus"]["observed"] = [0.] * 2
    elif mutation == "nan":
        data[0]["plus"]["model"][0] = float("nan")
    else:
        data[0]["plus"]["raw_actions"][0][0] = 1.1
    with pytest.raises(ValueError):
        audit.validate_endpoints(data, .1)


def test_physical_qp_label_not_misrepresented():
    result = audit.geometry(torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64), physical=True)
    assert result["objective_certificate_passed"] is True
    assert result["physical_labels_used"] is True
    assert result["deployed"] is False


def test_all_baseline_families_and_both_modes_present():
    assert len(audit.METHODS) == 8
    assert set(audit.METHODS) == {f + "_" + m for f in ("native_dispatch", "greedy", "zero", "cem") for m in ("replay", "tracking")}


def snapshot_fixture():
    packets = [{"packet": i, "snapshot": {"observed_root_rgb": {"sha256": str(i)}}, "root_rgb_sha256": str(i),
        "history_rgb_sha256": ["a", "b", "c"], "root_matches_history": True, "frames_match_history": True,
        "actions_match_history": True, "vector_matches_history": True} for i in range(1, 5)]
    return [{"repeat": i, "packets": copy.deepcopy(packets)} for i in (0, 1)], {"packets": copy.deepcopy(packets)}


def test_two_strict_snapshot_passes_bound_to_history():
    repeats, branch = snapshot_fixture()
    assert len(audit.snapshot_packets(repeats, branch)) == 4


@pytest.mark.parametrize("mutation", ["repeat", "missing", "state", "rgb", "flag"])
def test_snapshot_forgery_rejected(mutation):
    repeats, branch = snapshot_fixture()
    if mutation == "repeat":
        repeats[1]["repeat"] = 0
    elif mutation == "missing":
        repeats[1]["packets"].pop()
    elif mutation == "state":
        repeats[1]["packets"][0]["snapshot"]["qpos"] = [0., 0.]
    elif mutation == "rgb":
        branch["packets"][0]["root_rgb_sha256"] = "other"
    else:
        repeats[0]["packets"][0]["actions_match_history"] = False
    with pytest.raises(ValueError):
        audit.snapshot_packets(repeats, branch)
