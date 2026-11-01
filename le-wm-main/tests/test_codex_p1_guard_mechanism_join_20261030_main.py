import copy

import pytest

from experiments.spectral_reachability import codex_p1_guard_mechanism_join_20261030_main as join


def inputs():
    return [join.existing.gap.read(p, sha)[0] for p, sha in
            zip((join.VIEW, join.PREP, join.SAVED, join.PAIRED), join.SHAS, strict=True)]


def test_join_preserves_rejection_and_native_stop_at_horizon():
    output = join.build(*inputs())
    assert len(output["rows"]) == 24 and len(output["groups"]) == 8
    assert output["groups"][0]["selected_minus_zero"] is None
    assert output["groups"][2]["success"]["greedy"] is True
    assert output["rows"][8]["rgb_guard_feedback"]["status"] == "native_stopped"
    assert output["rows"][8]["rgb_guard_feedback"]["accepted_steps"] == 10


def test_join_rejects_tampered_prefix_or_endpoint():
    source = inputs()
    bad = copy.deepcopy(source)
    bad[1]["roster"][0]["prefix5_float32_hex"] = "00"
    with pytest.raises(ValueError):
        join.build(*bad)
    bad = copy.deepcopy(source)
    bad[2]["rows"][1]["endpoint_metrics"] = None
    with pytest.raises(ValueError):
        join.build(*bad)
