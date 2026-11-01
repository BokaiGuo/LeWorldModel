"""Numeric-only ordering/decomposition fixtures, not neural efficacy."""
import numpy as np
import pytest

from experiments.spectral_reachability.audit_reacher_d2p_metric_task_alignment import pair_orders, selected_attribution


def test_complete_pair_order_counts_and_reversal():
    values=list(range(17))
    same=pair_orders(values,values)
    assert same["pairs"]==same["agreement"]==136
    assert pair_orders(values,values[::-1])["reversal"]==136


def test_ties_are_not_falsely_counted_as_reversals():
    assert pair_orders([0.]*17,[0.]*17)["both_tie"]==136
    assert pair_orders([0.]*17,list(range(17)))["left_tie_only"]==136


def test_nonfinite_or_partial_rank_banks_rejected():
    with pytest.raises(ValueError): pair_orders([0.]*16,[0.]*17)
    with pytest.raises(ValueError): pair_orders([float("nan")]*17,[0.]*17)


def test_bounded_wrist_is_not_wrapped_in_error_decomposition():
    plan={"scores":{"decision":{"native_index":1,"greedy_index":0}}}
    observed={"bank_0":{"packets":[{"qpos":[3.,-2.]}]},"bank_1":{"packets":[{"qpos":[-3.,2.]}]}}
    out=selected_attribution(plan,observed,[-3.,-2.],[False,True])
    assert out["joint"]["metric"]["topology_residual"]==[0.,-4.]
    assert out["joint"]["metric"]["distance"]==4.
    np.testing.assert_allclose(sum(out["selected_minus_greedy_squared_error_by_joint"]),out["selected_minus_greedy_squared_error"])
