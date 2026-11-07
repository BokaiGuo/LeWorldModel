import copy

import pytest

from experiments.pc_bound_rgb_closed_loop_receipt_audit_20261107_codex import compare_receipts


def test_receipts_reject_input_drift_and_count_drift():
    value = [{"operation": "score", "inputs": {"pixels": {"sha256": "a"}}, "track_id": "primary_rgb"}]
    assert compare_receipts(value, copy.deepcopy(value)) == 1
    altered = copy.deepcopy(value)
    altered[0]["inputs"]["pixels"]["sha256"] = "b"
    with pytest.raises(ValueError, match="receipt mismatch"):
        compare_receipts(value, altered)
    with pytest.raises(ValueError, match="count mismatch"):
        compare_receipts(value, [])
