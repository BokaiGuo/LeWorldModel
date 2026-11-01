import copy
import json

import pytest

from experiments.spectral_reachability import codex_p1_cp5_physical_analysis_20261030_v1 as analysis
from experiments.spectral_reachability import codex_p1_cp5_physical_saved_20261030_v1 as saved


def test_real_receipt_has_24_distinct_labeled_rows():
    path = saved.run.ROOT/"results/codex_p1_cp5_physical_saved_20261030_v1_CODEX01.json"
    receipt = json.loads(path.read_bytes())
    result = analysis.analyze(receipt)
    assert len(result["rows"]) == 24
    assert all(row["status"] == "probe_completed" and row["returned_suffix"] == 10 for row in result["rows"])
    assert not any(row["endpoint_metrics"]["native_success"] for row in result["rows"])
    assert all(row["actual_resources"]["public_step"]["completed"] == 25 for row in result["rows"])
    bad = copy.deepcopy(receipt)
    bad["rows"][0]["role"] = "greedy"
    with pytest.raises(ValueError, match="identity"):
        analysis.analyze(bad)
