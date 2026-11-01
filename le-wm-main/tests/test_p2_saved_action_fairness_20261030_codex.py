import pytest

from experiments.spectral_reachability import p2_saved_action_fairness_20261030_codex as audit


def event(step=1, bits="0000000000000000"):
    return dict(kind="returned_transition", phase="plan", branch=0, step=step,
                actual_bits=bits, requested_bits=bits, actual_dtype="<f4")


def test_action_identity_and_missing_is_not_false():
    tables = {m: {1: "0000000000000000"} for m in audit.METHODS}
    tables["cem"][1] = "0000803f00000000"
    tables["gn"] = {}
    pairs = audit.compare_actions(tables, 2)
    assert pairs["h15__cem"]["identical_steps"] == 0
    assert pairs["h15__gradient"]["identical_steps"] == 1
    assert pairs["h15__gn"]["steps"][0]["same_action_bits"] is None
    assert pairs["h15__gradient"]["steps"][1]["same_action_bits"] is None


@pytest.mark.parametrize("rows,expected", [
    ([], 1), ([event(), event()], 2), ([event(2)], 1),
    ([dict(event(), requested_bits="0100000000000000")], 1),
    ([dict(event(), actual_dtype="<f8")], 1),
    ([event(bits="0000807f00000000")], 1),
    ([event(bits="x" * 16)], 1),
])
def test_fail_closed(rows, expected):
    with pytest.raises(ValueError):
        audit.action_table(rows, 0, expected)


def test_exact_bits_not_numeric_equality():
    tables = {m: {1: "0000000000000000"} for m in audit.METHODS}
    tables["cem"][1] = "0000008000000000"
    assert audit.compare_actions(tables, 1)["h15__cem"]["identical_steps"] == 0
