import copy
import pytest

from experiments.spectral_reachability import codex_e1_saved_ledger_bridge_20261030_v1 as m


def test_actual_receipts_bridge_without_invented_events():
    result = m.build(m.load_sources())
    assert result["status"] == "blocked_incomplete"
    assert result["equal_budget_certified"] is result["ranking_authorized"] is False
    assert len(result["slots"]) == 6
    for slot in result["slots"]:
        assert slot["ledger_preflight"]["structural_status"] == "incomplete"
        assert all(v is None for p in slot["ledger_input"]["phases"].values()
                   for v in p["measurements"].values())
        assert slot["receipt_counts_not_event_sets"]["controller"]["predict_api_calls"]["attempted"] == 144
        assert slot["receipt_counts_not_event_sets"]["replay"]["predictor_samples"]["completed"] == 4
    assert result["hi_four_slot_shared_unallocated"]["high_predictor"]["input_rows"] == 392
    assert result["slots"][1]["ledger_input"]["phases"]["inherited"]["scope"] == "shared_unallocated"


@pytest.mark.parametrize("change", [
    lambda s: s["prepare"].update(six_sha256="0"*64),
    lambda s: s["report"].update(prepare_sha256="0"*64),
    lambda s: s["controller"].update(report_sha256="0"*64),
    lambda s: s["controller"]["rows"][0].update(root=1),
    lambda s: s["controller"]["rows"][0].update(selection_sha256="0"*64),
    lambda s: s["targets"]["rows"][0].update(target_data_sha256="0"*64),
    lambda s: s["hi"]["rows"][0].update(high_target_sha256="0"*64),
    lambda s: s["controller"]["rows"][0]["controller_resources"]["predict_API"].update(attempted=True),
    lambda s: s["controller"]["rows"].pop(),
])
def test_mismatches_fail_closed(change):
    sources = copy.deepcopy(m.load_sources())
    change(sources)
    with pytest.raises((ValueError, KeyError)):
        m.build(sources)


def test_pin_mismatch_is_rejected(monkeypatch):
    sources = dict(m.SOURCES)
    sources["hi"] = (sources["hi"][0], "0" * 64)
    monkeypatch.setattr(m, "SOURCES", sources)
    with pytest.raises(ValueError, match="digest mismatch"):
        m.load_sources()
