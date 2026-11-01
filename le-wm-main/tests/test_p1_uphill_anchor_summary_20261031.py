import copy

import pytest

from experiments.spectral_reachability import p1_uphill_complete_pairing_20261030 as pairing
from experiments.spectral_reachability import p1_uphill_anchor_summary_20261031 as summary
from experiments.spectral_reachability import p1_uphill_pairing_csv_20261031 as table
from test_p1_uphill_complete_pairing_20261030 import inputs


def test_one_row_per_anchor_and_missing_endpoints_are_not_imputed():
    result = pairing.join(*inputs())
    long_rows = table.flatten(result)
    rows = summary.summarize(result)
    assert len(rows) == 24
    for row in rows:
        matching = [item for item in long_rows if item["anchor_index"] == row["anchor_index"]]
        values = [item["common_endpoint_object_position_error"] for item in matching
                  if item["common_endpoint_object_position_error"] is not None]
        assert row["common_probe_count"] == 42
        assert row["common_endpoint_object_position_error_observed_count"] == len(values)
        assert row["common_endpoint_object_position_error_missing_count"] == 42 - len(values)
        if values:
            assert row["common_endpoint_object_position_error_min"] == min(values)
            assert row["common_endpoint_object_position_error_max"] == max(values)
        else:
            assert row["common_endpoint_object_position_error_median"] is None
        assert row["multistart_status"] == matching[0]["multistart_status"]


def test_rejects_duplicate_probe():
    result = pairing.join(*inputs())
    result["common_suffix_pairs"][1] = copy.deepcopy(result["common_suffix_pairs"][0])
    with pytest.raises(ValueError, match="unique probe roster"):
        summary.summarize(result)
