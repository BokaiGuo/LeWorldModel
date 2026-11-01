import copy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_h10_integrated_table_20261030 as table

ROOT = Path(__file__).resolve().parents[1]


def sources():
    common = table.load(ROOT / "results/codex_p1_h10_paired_table_20261030_main_FULL01.json", table.TABLE_SHA)
    joined = table.load(ROOT / "results/codex_p1_h10_independent_join_20261030_v1_CODEX02.json", table.JOIN_SHA)
    return common, joined


def test_full_join_and_censoring():
    common, joined = sources()
    rows = table.build(common, joined)
    assert len(rows) == 1008
    assert len({(r["logical_pair"], r["role"]) for r in rows}) == 24
    for row in rows:
        if row["common_status"] != "probe_completed":
            assert row["common_endpoint_object_position_error"] is None


def test_bad_binding_fails_closed():
    common, joined = sources()
    changed = copy.deepcopy(joined)
    changed["rows"][0]["identity"]["condition"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        table.build(common, changed)
    changed = copy.deepcopy(joined)
    changed["rows"][0]["common_suffixes"][0]["endpoint_metrics"] = None
    with pytest.raises(ValueError, match="probe"):
        table.build(common, changed)
