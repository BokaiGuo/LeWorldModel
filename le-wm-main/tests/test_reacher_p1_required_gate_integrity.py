import pytest

from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner


@pytest.mark.parametrize("gates", [None, {}, {"cuda_only": True}, {"cuda_only": 1, "other": True},
                                  {"cuda_only": True, "other": False}])
def test_incomplete_or_failed_receipts_are_rejected(gates):
    with pytest.raises(ValueError, match="missing or failed"):
        runner._require_true_gates(gates, ("cuda_only", "other"), context="fixture")


def test_complete_boolean_gates_are_accepted():
    runner._require_true_gates({"cuda_only": True, "other": True}, ("cuda_only", "other"), context="fixture")


def test_extra_failed_gate_is_not_ignored():
    with pytest.raises(ValueError, match="failed gate"):
        runner._require_true_gates({"cuda_only": True, "other": True, "extra": False},
                                   ("cuda_only", "other"), context="fixture")
