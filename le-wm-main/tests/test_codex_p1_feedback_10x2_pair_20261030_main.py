import copy

import pytest

from experiments.spectral_reachability import codex_p1_feedback_10x2_pair_20261030_main as pair


def test_fixed_comparison_and_tamper_rejection():
    sources = [pair.util.gap.read(p, s)[0] for p, s in zip((pair.FEEDBACK, pair.PAIR), pair.SHAS, strict=True)]
    result = pair.build(*sources)
    assert result["summary"]["rows"] == 24
    assert result["summary"]["feedback_native_success"] == 0
    assert not result["equal_total_budget_certified"]
    bad = copy.deepcopy(sources)
    bad[0]["rows"][0]["role"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        pair.build(*bad)
    bad = copy.deepcopy(sources)
    bad[0]["rows"][0]["endpoint_metrics"] = None
    with pytest.raises(ValueError, match="endpoint"):
        pair.build(*bad)
