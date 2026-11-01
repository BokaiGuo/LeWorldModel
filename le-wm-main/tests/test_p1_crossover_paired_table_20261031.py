import copy

import pytest

from experiments.spectral_reachability import p1_crossover_paired_table_20261031 as table


def fixture():
    metrics = dict.fromkeys(table.METRICS, 1.0)
    rows = []
    for pair in range(8):
        for role in table.ROLES:
            for probe in table.PROBES:
                rows.append(dict(logical_pair=pair, root_slot=pair, condition="on", arm="a",
                                 role=role, probe=probe, probe_sha256=probe,
                                 status="probe_completed", accepted_suffix_steps=10,
                                 native_success=False, root=metrics.copy(),
                                 prefix=metrics.copy(), endpoint=metrics.copy()))
    return {"status": "saved_checks_passed", "rows": rows}


def test_no_imputation_for_missing_endpoint():
    data = fixture()
    data["rows"][1]["endpoint"] = None
    row = next(r for r in table.build(data)
               if r["logical_pair"] == 0 and r["role"] == "selected"
               and r["probe"] == "selected_feedback_tail")
    assert row["endpoint_object_position_error"] is None
    assert row["delta_vs_zero_object_position_error"] is None


def test_identity_and_duplicate_fail_closed():
    data = fixture()
    bad = copy.deepcopy(data)
    bad["rows"][4]["probe_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        table.build(bad)
    data["rows"][1] = copy.deepcopy(data["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        table.build(data)
