import copy

import pytest

from experiments.spectral_reachability import p3_raw_h3_panel_summary_20261031 as summary


@pytest.fixture
def inputs():
    manifest = summary.checked(summary.roster.source.MANIFEST, summary.roster.source.MANIFEST_SHA)
    dependence = summary.checked(summary.DEPENDENCE, summary.DEPENDENCE_SHA)
    receipts = []
    for i, (kind, digest) in enumerate(summary.RECEIPTS):
        path = summary.ROOT / f"results/p3_raw_h3_{kind}_pair_20261031_BATCH{i:02d}_RUN01.json"
        receipts.append(summary.checked(path, digest))
    return manifest, dependence, receipts


def test_complete_panel_and_mismatch_are_not_silently_merged(inputs):
    manifest, dependence, receipts = inputs
    result = summary.summarize(manifest, dependence, receipts)
    assert result["logical_pairs"] == 36 and result["unique_input_pairs"] == 18
    assert result["consistent_old_success_status_eligible"] == 13
    assert result["consistent_new_success_status_eligible"] == 13
    changed = copy.deepcopy(receipts)
    changed[0]["pairs"][0]["old_last_success"] = not changed[0]["pairs"][0]["old_last_success"]
    mismatch = summary.summarize(manifest, dependence, changed)
    assert mismatch["repeat_mismatch_pairs"] == result["repeat_mismatch_pairs"] + 1
    assert mismatch["exact_repeat_pairs"] == result["exact_repeat_pairs"] - 1


def test_rejects_missing_or_wrong_pair_identity(inputs):
    manifest, dependence, receipts = inputs
    with pytest.raises(ValueError, match="roster source contract"):
        summary.summarize(manifest, dependence, receipts[:-1])
    changed = copy.deepcopy(receipts)
    changed[0]["pairs"][0]["candidate"] = "wrong"
    with pytest.raises(ValueError, match="pair identity"):
        summary.summarize(manifest, dependence, changed)
