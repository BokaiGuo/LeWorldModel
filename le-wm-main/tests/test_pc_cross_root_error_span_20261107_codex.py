import numpy as np

from experiments.pc_cross_root_error_span_20261107_codex import assess, span


def outcomes():
    return [{"final_distance": float(i // 8), "ever": i < 8, "prefix_success": False} for i in range(64)]


def test_shift_invariance_and_margin_bound():
    p = np.broadcast_to(np.arange(8)[:, None] * 10., (8, 8)).copy()
    a = p + np.linspace(-1, 1, 64).reshape(8, 8)
    assert np.isclose(span(p, a), span(p + 1000, a))
    result = assess(p, a, outcomes(), threshold=2.)
    assert result["triggered"] and result["empirical_span_covered"] and result["encoding_best_match"]
    assert result["selected_chain_success"]


def test_observed_out_of_calibration_does_not_claim_theorem():
    p = np.broadcast_to(np.arange(8)[:, None] * 10., (8, 8)).copy()
    a = p.copy()
    a[1, :] -= 100
    result = assess(p, a, outcomes(), threshold=2.)
    assert result["triggered"] and not result["empirical_span_covered"]
    assert not result["encoding_best_match"]
