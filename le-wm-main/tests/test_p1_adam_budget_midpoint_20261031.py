import copy

import pytest

from experiments.spectral_reachability import p1_adam_budget_midpoint_cuda_20261031 as run
from experiments.spectral_reachability import p1_adam_budget_midpoint_analysis_20261031 as analysis


def test_bound_panel():
    entries, stopped, sources = run.inputs()
    assert (len(entries), len(stopped)) == (64, 8)
    assert len(sources) >= 281


def test_analysis_rejects_corrupt_latent_and_duplicate():
    data = run.terminal.pair.pairing.read(analysis.SOURCE, analysis.SOURCE_SHA)
    result = analysis.analyze(data)
    assert result["summary"]["5-10"]["pairs"] == 21
    changed = copy.deepcopy(data)
    changed["rows"][0]["refresh_latent"][0] += 1
    with pytest.raises(Exception):
        analysis.analyze(changed)
    changed = copy.deepcopy(data)
    changed["rows"][1] = changed["rows"][0]
    with pytest.raises(Exception):
        analysis.analyze(changed)
