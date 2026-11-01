import pytest

from experiments.spectral_reachability import p1_four_arm_winner_root1_20261031 as subject


def test_root1_store_rewrites_only_expected_namespace():
    assert subject.Root1Store.name("s0_native_b05_winner") == "s1_native_b05_winner"
    with pytest.raises(ValueError):
        subject.Root1Store.name("s2_native_b05_winner")


def test_protocol_includes_winner_budget_and_known_root():
    subject.configure()
    contract = subject.protocol()
    assert contract["root_slot"] == 1
    assert contract["extra_policy"]["native"] == "model_winner"
    assert contract["extra_policy"]["random"] == "model_winner"
    assert contract["extra_policy"]["structured"] == "none"
    assert contract["fairness_proven"] is False
    assert contract["independent_confirmation"] is False
