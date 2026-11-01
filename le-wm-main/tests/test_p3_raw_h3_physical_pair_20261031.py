import copy

import pytest

from experiments.spectral_reachability import p3_raw_h3_physical_pair_20261031 as pair
from test_p3_raw_h3_physical_saved_20261031 import completed


def test_pairs_require_real_step15_and_keep_missing_endpoints(completed):
    _, report, _, _, _ = completed
    rows = pair.pair_rows(report)
    assert len(rows) == 3
    assert all(row["step15_exact"] is True for row in rows)
    assert rows[0]["new_minus_old_endpoint"]["native_distance"] == 0
    changed = copy.deepcopy(report)
    changed["rows"][1]["checkpoints"][3]["snapshot"] = {"state": [1]}
    assert pair.pair_rows(changed)[0]["step15_exact"] is False
    assert pair.pair_rows(changed)[0]["new_minus_old_endpoint"] is None
    changed = copy.deepcopy(report)
    changed["rows"][1]["full_horizon"] = False
    changed["rows"][1]["own_returned"] = 10
    changed["rows"][1]["checkpoints"] = changed["rows"][1]["checkpoints"][:3]
    assert pair.pair_rows(changed)[0]["step15_exact"] is None
    assert pair.pair_rows(changed)[0]["new_endpoint"] is None
    changed = copy.deepcopy(report)
    changed["rows"][1]["full_horizon"] = False
    changed["rows"][1]["status"] = "native_success"
    assert pair.pair_rows(changed)[0]["new_endpoint"] is not None
    changed["rows"][1]["status"] = "failed"
    assert pair.pair_rows(changed)[0]["new_endpoint"] is None


def test_pair_identity_mismatch_is_rejected(completed):
    _, report, _, _, _ = completed
    changed = copy.deepcopy(report)
    changed["rows"][1]["selected_plan"]["candidate"] = "different"
    with pytest.raises(ValueError, match="pair identity"):
        pair.pair_rows(changed)
