import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability.p2_exclusive_resource_gap_20261101 import audit


@pytest.fixture
def saved():
    path = Path(__file__).resolve().parents[1] / "results/p2_saved_fair_pair_20261029_MAIN02.json"
    return json.loads(path.read_text())


def test_real_saved_census(saved):
    result = audit(saved)
    assert (result["pairs"], result["comparable"]) == (18, 16)
    assert result["P2_admission"] is False
    assert all(row["axes"]["FLOPs"] == "unknown" for row in result["joint_vs_composition"])
    assert all(row["axes"]["predictor_samples"] == "different" for row in result["joint_vs_composition"])


@pytest.mark.parametrize("mutation", ["duplicate", "missing_axis", "inconsistent", "bool"])
def test_rejects_bad_data(saved, mutation):
    data = copy.deepcopy(saved)
    if mutation == "duplicate":
        data["pairs"][1] = copy.deepcopy(data["pairs"][0])
    elif mutation == "missing_axis":
        del data["pairs"][0]["resources_left"]["VJP"]
    elif mutation == "inconsistent":
        data["pairs"][0]["resources_left"]["VJP"] += 1
    else:
        data["pairs"][0]["resources_left"]["VJP"] = True
    with pytest.raises(ValueError):
        audit(data)
