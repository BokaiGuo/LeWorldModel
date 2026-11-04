import json

import pytest

from experiments.spectral_reachability import p3_point30_physical_prepare_20261102 as module
from experiments.spectral_reachability import p3_point30_physical_saved_20261102 as saved


def test_saved_plan_reconstructs_all_point30_winners():
    plan = module.material()
    path = module.ROOT / "results/p3_point30_physical_20261102_PREP_03.json"
    assert json.loads(path.read_text()) == plan
    assert [(x["arm"], x["root_slot"], x["candidate_index"]) for x in plan["plans"]] == [
        ("signal", 0, 0), ("signal", 1, 4), ("reverse", 0, 16), ("reverse", 1, 4)
    ]
    assert plan["physical_execution"] is False
    assert plan["independent_outcome"] is False
    assert plan["equal_total_compute"] is False


def test_saved_physics_validates_and_rejects_wrong_report_hash():
    plan = module.ROOT / "results/p3_point30_physical_20261102_PREP_03.json"
    run = module.ROOT / "results/p3_point30_physical_20261102_RUN_01"
    plan_sha = "152e0e6ab143d26a48cde8d99b84ab7815d9aaffafa823eeac8cd66c34e94dda"
    report_sha = "0343a14644f74de19d328ea9d11398e9cfc32c0166482d4bd0e775173f217f01"
    result = saved.consume(plan, plan_sha, run, report_sha)
    assert result["status"] == "saved_physical_records_verified"
    assert [row["returned"] for row in result["branches"]] == [10] * 4
    with pytest.raises(ValueError, match="report SHA"):
        saved.consume(plan, plan_sha, run, "0" * 64)
