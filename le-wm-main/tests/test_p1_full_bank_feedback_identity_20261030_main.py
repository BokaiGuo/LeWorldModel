import json

import pytest

from experiments.spectral_reachability import p1_full_bank_feedback_identity_20261030_main as join


def load():
    return {name: json.loads((join.ROOT / path).read_bytes())
            for name, (path, _) in join.SOURCES.items()}


def test_all_eight_prefixes_match_and_censor_is_preserved():
    result = join.build(**load())
    assert len(result["rows"]) == 8
    assert all(row["prefix_identical_to_prior_adam"] for row in result["rows"])
    assert result["rows"][2]["controls"]["zero"]["independent_adam"]["horizon_metrics"] is None
    assert not result["equal_budget_certified"]


@pytest.mark.parametrize("field", ["action", "pixel", "snapshot", "task"])
def test_rejects_mismatched_prefix_identity(field):
    data = load()
    row = data["physical"]["rows"][5]
    cp = next(x for x in row["checkpoints"] if x["step"] == 5)
    if field == "action":
        row["plan"]["raw15_float32_hex"] = "ab" + row["plan"]["raw15_float32_hex"][2:]
    elif field == "pixel":
        cp["pixel_sha256"] = "tampered"
    elif field == "snapshot":
        cp["snapshot"]["state"][0] += 1
    else:
        row["task_id"] = "tampered"
    with pytest.raises(ValueError, match="identity"):
        join.build(**data)
