import json

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_cem32_decode_parity_20261102 as audit


def test_compare_counts_float32_bits_and_distance():
    left = np.array([1.0, 0.0], np.float32)
    right = np.array([np.nextafter(np.float32(1), np.float32(2)), 0.0], np.float32)
    result = audit.compare(left, right)
    assert result == {"bitwise_equal": False, "unequal_elements": 1, "max_abs": 2**-23}
    assert audit.compare(left, left)["bitwise_equal"]


def test_compare_rejects_invalid_values():
    with pytest.raises(Exception):
        audit.compare([float("nan")], [0])
    with pytest.raises(Exception):
        audit.compare([0, 1], [0])


def test_saved_values_recompute_and_match_physical_plan():
    path = audit.ROOT / "results/reacher_cem32_decode_parity_RUN_20261102_v1.json"
    if not path.exists():
        pytest.skip("CUDA run not present")
    result = json.loads(path.read_text())
    physical = json.loads(audit.rgb.PLAN.read_text())
    branches = [b for root in physical["roots"] for b in root["branches"]]
    assert len(result["rows"]) == len(branches) == 6
    for row, branch in zip(result["rows"], branches, strict=True):
        executed = np.asarray(branch["raw_actions"], np.float32)[5:]
        assert row["cpu_vs_cuda_raw"] == audit.compare(row["cpu_raw"], row["cuda_raw"])
        assert row["cpu_vs_cuda_normalized"] == audit.compare(row["cpu_normalized"], row["cuda_normalized"])
        assert row["cpu_vs_executed"] == audit.compare(row["cpu_raw"], executed)
        assert row["cuda_vs_executed"] == audit.compare(row["cuda_raw"], executed)
        assert row["cpu_vs_executed"]["bitwise_equal"]
