import copy
import json

import pytest

from experiments.spectral_reachability import codex_p1_adam_paired_join_20261030_main as join


def inputs():
    return {key: json.loads((join.ROOT / path).read_text())
            for key, (path, _) in join.SOURCES.items()}


def test_complete_censor_and_zero_reference():
    result = join.build(**inputs())
    assert len(result["rows"]) == 24
    stopped = [r["adam_independent_suffix"] for r in result["rows"]
               if r["adam_independent_suffix"]["status"] == "native_stopped"]
    assert len(stopped) == 5
    assert all(r["horizon_metrics"] is None and r["endpoint_minus_own_zero_suffix"] is None for r in stopped)
    assert all(r["last_observed_metrics"]["native_success"] for r in stopped)
    for row in result["rows"]:
        if row["identity"]["role"] == "zero" and row["adam_independent_suffix"]["horizon_metrics"] is not None:
            assert set(row["adam_independent_suffix"]["endpoint_minus_zero_prefix_independent_endpoint"].values()) == {0}


@pytest.mark.parametrize("target", ["prefix", "action", "identity"])
def test_tampered_bindings_rejected(target):
    data = inputs()
    data = copy.deepcopy(data)
    if target == "prefix":
        data["prepare"]["roster"][0]["prefix5_float32_hex"] = "00"
    elif target == "action":
        data["saved"]["rows"][0]["raw10_sha256"] = "wrong"
    else:
        data["saved"]["rows"][0]["role"] = "zero"
    with pytest.raises(ValueError):
        join.build(**data)
