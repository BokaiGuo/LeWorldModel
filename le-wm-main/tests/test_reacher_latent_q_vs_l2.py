import pytest

from experiments.spectral_reachability.audit_reacher_latent_q_vs_l2 import compare


def test_order_comparison_preserves_signs_and_ties():
    assert compare(-1.0, -2.0) == "agreement"
    assert compare(-1.0, 2.0) == "reversal"
    assert compare(0.0, 2.0) == "tie"
    with pytest.raises(ValueError, match="nonfinite"):
        compare(float("nan"), 1.0)
