import copy
import json

import numpy as np
import pytest

from experiments import pc_response_diverse_beam_trace_attribution_20261103_codex as diag


def test_sealed_attribution_and_refinement_start():
    if not diag.OUTPUT.exists():
        pytest.skip("attribution artifact unavailable")
    data = json.loads(diag.OUTPUT.read_text())
    source = json.loads(diag.exp.PLANS.read_text())
    assert all(diag.exp.prior.sha(p) == digest for p, digest in data["sources"].items())
    original_rows = [row for row in source["rows"] if row["arm"] == "response_diverse"]
    assert len(data["rows"]) == len(original_rows) == 24
    for row, receipt in zip(original_rows, data["rows"], strict=True):
        assert diag.analyze(row) == receipt
        assert abs(receipt["selection_gap"]) < 1e-12
        batches = [b for b in row["trace"] if np.asarray(b["requested"]).shape[1:] == (15, 2)]
        suffix_batches = sum((p["samples"] + 11) // 12 for p in row["result"]["programs"])
        input_index = receipt["refinement_input_index"]
        np.testing.assert_array_equal(batches[suffix_batches]["requested"][0],
                                      row["result"]["programs"][input_index]["chain"])
    corrupted = copy.deepcopy(original_rows[0])
    corrupted["result"]["programs"][0]["terminal_score"] += .1
    with pytest.raises(AssertionError):
        diag.analyze(corrupted)
