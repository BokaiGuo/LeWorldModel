from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_diag32_fixed_physical_20261101_exclusive as subject


def test_pinned_unique_candidate_and_bounds():
    report = json.loads((subject.DIAG_DIR / "report.json").read_bytes())
    commands, selected = subject.fixed_selection(report)
    assert (selected["candidate"], selected["iteration"]) == (4, 13)
    assert np.asarray(commands).shape == (15, 2)
    assert np.max(np.abs(commands)) <= 1
    altered = deepcopy(report)
    altered["rows"][0]["selected"]["iteration"] = 12
    with pytest.raises(ValueError):
        subject.fixed_selection(altered)
    altered = deepcopy(report)
    extra = altered["rows"][1]["evaluations"][0]
    extra["q1"] = altered["rows"][1]["root_mse"] + 1
    extra["q3"] = altered["rows"][1]["root_mse"] - 0.1
    with pytest.raises(ValueError):
        subject.fixed_selection(altered)


def test_contract_source_and_normalization():
    contract = subject.build()
    a = np.asarray(contract["commands"], np.float32)
    mean = np.asarray(contract["action_mean"], np.float32)
    std = np.asarray(contract["action_std"], np.float32)
    assert np.isfinite((a - mean) / std).all()
    assert contract["physical_caps"]["public_step"] >= 25
    assert contract["physical_future_used_for_selection"] is False
