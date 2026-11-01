from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_backtrack_physical_contract_20261030 as contract


@pytest.fixture
def source():
    path = Path(__file__).resolve().parents[1] / "results/p1_backtrack_physical_source_join_20261030_MAIN02.json"
    if not path.is_file():
        pytest.skip("source join unavailable")
    return json.loads(path.read_text())


def test_all_sixteen_rows_and_explicit_physical_resource_bounds(source):
    plans = contract.plans(source)
    assert len(plans) == 16
    assert [(p["slot"], p["arm"], p["role"]) for p in plans] == contract.ORDER
    caps = contract.contract()["caps"]
    assert caps["public_step"] == 16 * (10 + 15)
    assert caps["physics_step"] == 2 * caps["reset"] + 10 * caps["public_step"]
    assert caps["explicit_render"] == 16 * (11 + 16) * 3
    assert contract.contract()["P1_admission"] is False


def test_rejects_reordering_and_unbound_native_root(source):
    bad = deepcopy(source)
    bad["rows"][0], bad["rows"][1] = bad["rows"][1], bad["rows"][0]
    with pytest.raises(ValueError, match="mismatch"):
        contract.plans(bad)
    bad = deepcopy(source)
    bad["rows"][0]["physical_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        contract.plans(bad)
