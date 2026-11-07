import numpy as np
import pytest

from experiments.pc_empirical_opportunity_bound_20261107_codex import evaluate


def test_margin_certifies_actual_encoding_but_not_physical_order():
    prediction = np.broadcast_to(np.arange(8)[:, None], (8, 8)).copy()
    encoding = prediction + 0.1
    outcomes = [{"final_distance": float(8-i), "ever": i == 7, "prefix_success": False}
                for i in range(8) for _ in range(8)]
    result = evaluate(prediction, encoding, outcomes)
    assert all(x["certified"] and x["encoded_same_sign"] for x in result["pairs"])
    assert not any(x["physical_same_sign"] for x in result["pairs"])


def test_no_certification_when_margin_is_small():
    p = np.zeros((8, 8))
    q = np.ones((8, 8))
    outcomes = [{"final_distance": 1., "ever": False, "prefix_success": False} for _ in range(64)]
    assert not any(x["certified"] for x in evaluate(p, q, outcomes)["pairs"])
    with pytest.raises(ValueError):
        evaluate(p[:, :7], q, outcomes)
