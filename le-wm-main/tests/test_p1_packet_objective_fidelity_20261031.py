import copy
import json

import pytest

from experiments.spectral_reachability import p1_packet_objective_fidelity_analysis_20261031 as m


def test_real_latents_recompute_and_tampering_is_rejected():
    value = json.loads(m.INPUT.read_bytes())
    result = m.analyze(value)
    assert len(result["pairs"]) == 8
    assert sum(x["comparisons"][0]["model_rgb_reversal"] for x in result["pairs"]) == 3
    changed = copy.deepcopy(value)
    changed["rows"][0]["packets"][0]["predicted_latent"][0] += 1
    with pytest.raises(ValueError):
        m.analyze(changed)
    changed = copy.deepcopy(value)
    changed["rows"][0]["role"] = "zero"
    with pytest.raises(ValueError):
        m.analyze(changed)
