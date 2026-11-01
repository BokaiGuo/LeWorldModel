import copy

import pytest

from experiments.spectral_reachability import p1_adam_terminal_rgb_analysis_20261031 as analysis


def test_saved_panel_and_latent_integrity():
    data = analysis.pair.pairing.read(analysis.SOURCE, analysis.SOURCE_SHA)
    result = analysis.analyze(data)
    assert result["summary"]["5-10"]["pairs"] == 21
    assert result["summary"]["5-10"]["predicted_vs_rgb_opposite"] == 9
    altered = copy.deepcopy(data)
    altered["rows"][0]["actual_latent"][0] += 1
    with pytest.raises(Exception):
        analysis.analyze(altered)
