import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p5_root_stratified_saved_codex_20261101_exclusive as audit

SOURCE = Path(__file__).resolve().parents[1] / "results/p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1_MAIN01.json"


@pytest.fixture
def panel():
    return json.loads(SOURCE.read_bytes())


def test_real_panel_strata_and_horizon_one(panel):
    result = audit.summarize(panel)
    assert result["branch_count"] == 164
    assert len(result["groups"]) == 120
    assert all(x["tie"] == 41 for x in result["groups"] if x["horizon"] == 1)
    assert result["P5_admission"] is False


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_error", "wrong_time", "nonfinite", "wrong_task"])
def test_reject_corruption(panel, mutation):
    panel = copy.deepcopy(panel)
    if mutation == "duplicate":
        panel["rows"][1]["index"] = panel["rows"][0]["index"]
    elif mutation == "wrong_error":
        panel["rows"][0]["steps"][0]["open_absolute_error"][0] += 1
    elif mutation == "wrong_time":
        panel["rows"][0]["steps"][0]["plan_step"] = 10
    elif mutation == "nonfinite":
        panel["rows"][0]["steps"][0]["physical"][0] = float("nan")
    else:
        panel["rows"][0]["task_id"] = "different"
    with pytest.raises(ValueError):
        audit.summarize(panel)
