import copy
import json

import pytest

from experiments.spectral_reachability import p3_rgb_fallback_h3_summary_20261031 as summary


@pytest.fixture(scope="module")
def sample():
    expected = json.loads(summary.PREDICTED.read_bytes())["rows"]
    rows = []
    for item in expected:
        rows.append({"identity": item["identity"], "pair_key": item["pair_key"],
                     "old_endpoint": item["old_endpoint"], "new_endpoint": item["endpoint"],
                     "new_minus_old": item["delta_vs_old"],
                     "old_native_success": item["old_endpoint"]["native_success"],
                     "new_native_success": item["success"],
                     "old_suffix_selection": "cached_fallback" if item["origin"] ==
                     "cached_old_counterfactual" else "planned",
                     "new_suffix_selection": "cached_fallback" if item["origin"] ==
                     "cached_old_counterfactual" else "planned"})
    return rows, expected


def test_complete_roster_and_dedup(sample):
    result = summary.summarize(*sample)
    assert result["fixed_pairs"] == 52
    assert result["unique_action_pairs"] == 26
    assert result["offline_endpoint_pairs_exact"] == 52


def test_missing_identity_and_duplicate_disagreement_are_rejected(sample):
    rows, expected = copy.deepcopy(sample)
    rows[0]["pair_key"] = "different"
    with pytest.raises(ValueError, match="action pair"):
        summary.summarize(rows, expected)
    rows, expected = copy.deepcopy(sample)
    first = rows[0]
    other = next(r for r in rows[1:] if r["pair_key"] == first["pair_key"])
    other["new_native_success"] = not other["new_native_success"]
    with pytest.raises(ValueError, match="repeated action"):
        summary.summarize(rows, expected)
