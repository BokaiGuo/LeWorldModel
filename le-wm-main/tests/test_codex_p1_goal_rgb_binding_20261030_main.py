import pytest

from experiments.spectral_reachability.codex_p1_goal_rgb_binding_20261030_main import validate_identity


def test_matching_identity():
    row = {"index": 2, "logical_pair": 0, "role": "greedy"}
    item = {"index": 2, "identity": {"logical_pair": 0, "role": "greedy"}}
    assert validate_identity(2, row, item) == item["identity"]


@pytest.mark.parametrize("field,value", [("index", 1), ("logical_pair", 1), ("role", "zero")])
def test_mismatched_identity_rejected(field, value):
    row = {"index": 2, "logical_pair": 0, "role": "greedy"}
    item = {"index": 2, "identity": {"logical_pair": 0, "role": "greedy"}}
    row[field] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_identity(2, row, item)
