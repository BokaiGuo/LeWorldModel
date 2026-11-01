import json

import pytest

from experiments.spectral_reachability import p1_h10_multimetric_pair_20261030 as audit


@pytest.fixture
def payload():
    if not audit.SOURCE.is_file():
        pytest.skip("sealed H10 source unavailable")
    return json.loads(audit.SOURCE.read_bytes())


def test_all_roster_and_metrics(payload):
    result = audit.analyze(payload)
    assert len(result["rows"]) == 6
    assert all(row["matched_nonzero_suffixes"] == 40 for row in result["rows"])
    assert all(set(row["selected_minus_reference_absolute"]) == set(audit.METRICS)
               for row in result["rows"])
    assert not result["P1_admission"]


def test_tampered_pair_and_censor_are_rejected(payload):
    control = next(row for row in payload["rows"] if row["logical_pair"] == 6 and row["role"] == "zero")
    item = control["common_suffixes"][1]["item"]
    item["endpoint"]["object_position_error"] += 1
    with pytest.raises(ValueError, match="paired difference binding"):
        audit.analyze(payload)
    item["endpoint"]["object_position_error"] -= 1
    item["observation"]["censor"] = "stopped"
    with pytest.raises(ValueError, match="probe completion"):
        audit.analyze(payload)
