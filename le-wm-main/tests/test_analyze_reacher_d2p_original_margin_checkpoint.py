"""Independent algebra fixtures only; no predictive efficacy evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_original_margin_checkpoint import check_cell


def fixture():
    return {"vectors": {"prediction": [1.] * 192, "prediction_zero": [0.] * 192,
        "observed": [-1.] * 192, "observed_zero": [0.] * 192, "goal": [2.] * 192},
        "costs": {"prediction": 1., "prediction_zero": 4., "observed": 9., "observed_zero": 4.},
        "candidate_latent_l2": float(np.sqrt(768)), "zero_latent_l2": 0.,
        "decomposition": {"predicted_margin": -3., "observed_margin": 5., "margin_error": 8.,
            "zero_successor_bias_contribution": 0., "response_linear_error_contribution": 8.,
            "response_quadratic_error_contribution": 0.}}


def test_numpy_decomposition_independent_of_torch():
    assert check_cell(fixture())["margin_error"] == 8.


@pytest.mark.parametrize("kind", ["missing_vector", "shape", "nan", "decomposition", "cost", "l2"])
def test_tampered_vector_evidence_rejected(kind):
    cell = copy.deepcopy(fixture())
    if kind == "missing_vector":
        del cell["vectors"]["goal"]
    elif kind == "shape":
        cell["vectors"]["goal"].pop()
    elif kind == "nan":
        cell["vectors"]["prediction"][0] = float("nan")
    elif kind == "decomposition":
        cell["decomposition"]["response_linear_error_contribution"] = 0.
    elif kind == "cost":
        cell["costs"]["observed"] = 0.
    else:
        cell["candidate_latent_l2"] = 0.
    with pytest.raises(ValueError):
        check_cell(cell)
