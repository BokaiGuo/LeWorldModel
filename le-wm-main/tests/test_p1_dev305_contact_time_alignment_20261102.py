import json

import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_contact_time_alignment_saved_20261102 import summarize


def test_ranking_reversal_and_step_index():
    prediction = np.zeros((4, 192))
    prediction[:, 0] = [1, 2, 3, 4]
    reality = np.repeat(prediction[:, None, :], 5, axis=1)
    reality[:, 0, 0] = [4, 3, 2, 1]
    rows = summarize(prediction, reality)
    json.dumps(rows, allow_nan=False)
    assert rows[0]["step"] == 1
    assert rows[0]["reversed_pairs"] == rows[0]["comparable_pairs"] == 6
    assert rows[0]["model_choice"] == 0
    assert rows[0]["real_choice"] == 3
    assert rows[-1]["reversed_pairs"] == 0
    assert rows[-1]["forecast_vs_real_mse_mean"] == 0


def test_ties_are_excluded_and_invalid_input_is_rejected():
    prediction = np.zeros((4, 192))
    reality = np.zeros((4, 5, 192))
    assert summarize(prediction, reality)[0]["comparable_pairs"] == 0
    reality[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        summarize(prediction, reality)
