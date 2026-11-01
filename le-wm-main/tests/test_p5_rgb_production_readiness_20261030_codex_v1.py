import copy
import sys

import pytest

from experiments.spectral_reachability import p5_rgb_production_readiness_20261030_codex_v1 as prep


@pytest.fixture(scope="module")
def panel():
    return prep.wiring.runner.load_prepared()


def test_actual_saved_panel_and_no_model(panel):
    data = prep.census(panel)
    assert data["branch_denominator"] == 164 and data["pair_denominator"] == 80
    assert data["conditions"]["off"]["branches"] == data["conditions"]["on"]["branches"] == 82
    assert data["conditions"]["off"]["actions"] == data["conditions"]["on"]["actions"] == 410
    assert data["off_on_input_hash_differences"] == {"previous_rgb": 82, "current_rgb": 82}
    assert "torch" not in sys.modules


@pytest.mark.parametrize("kind", ["branch", "action", "pair", "history"])
def test_tampering_fails(panel, kind):
    value = copy.deepcopy(panel)
    if kind == "branch":
        value["rows"].pop()
    elif kind == "action":
        value["rows"][0]["actions"][0][0] += 1
    elif kind == "pair":
        value["pairs"][0]["coord"] = 9
    else:
        value["rows"][1]["current_rgb"]["model_input_chw_float32_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        prep.census(value)


def test_resource_counts_real_sources():
    report = prep.prepare(1024)
    assert report["planned_operation_caps"]["maximum_neural_interface_calls"] == 984
    assert report["logical_bytes"]["model_parameters_float32"] == 375368
    assert report["logical_bytes"]["images_per_branch_float32"] == 98304
    assert "CUDA runtime and allocator memory" in report["unknown_not_bounded"]
    assert report["gpu_calls"] == report["neural_forward_calls"] == report["physics_calls"] == 0
    assert "torch" not in sys.modules
