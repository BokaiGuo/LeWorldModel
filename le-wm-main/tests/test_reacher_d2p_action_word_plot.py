import json

import numpy as np
import pytest

from experiments.spectral_reachability.plot_reacher_d2p_action_words import margin_counts


def test_counts_json_safe_and_neutral_not_counted_as_success():
    result = margin_counts(np.array([0., 1., -1., .1, 0.]), np.array([0., 2., 1., 0., 1.]))
    assert json.loads(json.dumps(result)) == {"strict_margin_agreements": 1, "strict_margin_reversals": 1, "ties": 2}


@pytest.mark.parametrize("predicted,observed", [([0.], [0.]), ([0., 1.], [0.]), ([0., float("nan")], [0., 1.])])
def test_invalid_margins_rejected(predicted, observed):
    with pytest.raises(ValueError):
        margin_counts(predicted, observed)
