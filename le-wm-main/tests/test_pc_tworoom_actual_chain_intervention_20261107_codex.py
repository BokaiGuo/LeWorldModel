import numpy as np

from experiments.pc_tworoom_actual_chain_intervention_20261107_codex import make_prefixes
from experiments.pc_tworoom_actual_chain_intervention_analysis_20261107_codex import successes


def row(n):
    trace = [{"action": [0.2, -0.3]} for _ in range(n)]
    decisions = [{"actions": [[0.4, -0.5]] * 25}]
    return {"episode": 9000, "arms": {"effect": {"trace": trace}, "lookahead": {"decisions": decisions}}}


def test_prefixes_bound_to_executed_chain_and_lookahead_plan():
    chain, prefixes = make_prefixes(row(8))
    np.testing.assert_allclose(prefixes["actual"], chain[:5])
    np.testing.assert_allclose(prefixes["lookahead"], np.tile([0.4, -0.5], (5, 1)))
    np.testing.assert_array_equal(prefixes["zero"], 0)
    np.testing.assert_array_equal(prefixes["random"], make_prefixes(row(8))[1]["random"])
    assert np.abs(prefixes["random"]).max() <= 1


def test_short_actual_trace_keeps_task_without_fabricated_prefix():
    chain, prefixes = make_prefixes(row(3))
    assert len(chain) == 3
    assert prefixes is None


def test_success_accounting_distinguishes_prefix_and_replanned():
    failed, successful = {"success": False}, {"success": True}
    arm = {"prefix_trace": [failed], "common_tail_trace": [successful], "replanned_trace": [failed]}
    assert successes(arm) == (False, True, False)
    arm["prefix_trace"] = [successful]
    assert successes(arm) == (True, True, True)
