import pytest

from experiments.spectral_reachability.audit_reacher_d2p_refined_selector import validated_rows


def report():
    return {"status": "completed", "gates": {"ok": True}, "sealed_protocol": {"root_seeds": [2, 1]},
        "rows": [{"seed": 1}, {"seed": 2}]}


def test_reference_is_joined_by_identity_not_array_position():
    assert set(validated_rows(report())) == {1, 2}


@pytest.mark.parametrize("case", ["duplicate", "empty", "missing", "failed"])
def test_invalid_reference_fails_closed(case):
    value = report()
    if case == "duplicate":
        value["rows"][1]["seed"] = 1
    elif case == "empty":
        value["rows"] = []
        value["sealed_protocol"]["root_seeds"] = []
    elif case == "missing":
        value["rows"].pop()
    else:
        value["gates"]["ok"] = False
    with pytest.raises(ValueError):
        validated_rows(value)
