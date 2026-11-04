import copy
import json

import numpy as np
import pytest

from experiments import pc_rgb_successor_exact_cem_trace_reconstruction_20261103_codex as audit


def test_receipt_and_reconstruction_detects_tampering():
    if not audit.OUTPUT.exists():
        pytest.skip("reconstruction artifact unavailable")
    receipt = json.loads(audit.OUTPUT.read_text())
    assert all(audit.source.prior.sha(path) == digest for path, digest in receipt["sources"].items())
    original = json.loads(audit.source.PLANS.read_text())
    assert len(receipt["rows"]) == 144
    assert sum(len(row["batches"]) for row in receipt["rows"]) == 4608
    for root_index in (0, 11, 23):
        row = original["rows"][root_index]
        for run_index in (0, 4, 5):
            reconstructed = audit.reconstruct(row, row["runs"][run_index])
            assert reconstructed == receipt["rows"][root_index * 6 + run_index]
    row = original["rows"][0]
    corrupted = copy.deepcopy(row["runs"][0])
    corrupted["trace"][1]["actions"][1][-1][0] += 0.001
    with pytest.raises(AssertionError):
        audit.reconstruct(row, corrupted)
    assert np.isclose(receipt["rows"][4]["batches"][0]["scale_median"], 1 / 3)
