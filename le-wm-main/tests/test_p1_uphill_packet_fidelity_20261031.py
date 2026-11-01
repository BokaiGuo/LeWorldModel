import copy

import pytest

from experiments.spectral_reachability import p1_uphill_packet_fidelity_20261031 as m


def test_all_sealed_rows_include_observed_first_packet_and_no_imputed_h10():
    entries, _ = m.inputs()
    assert len(entries) == 24
    assert sum(5 in entry["images"] for entry in entries) == 24
    assert sum(10 in entry["images"] for entry in entries) == 22
    assert all((10 in entry["images"]) == (entry["steps"] >= 10) for entry in entries)


def test_frame_binding_rejects_wrong_state_and_overlay():
    row = {"trajectory": [{"phase": "probe", "step": 5, "overlay_index": 0,
                           "condition": "off", "state": [1]}],
           "responses": [{"state": [0]} for _ in range(4)] + [{"state": [1]}]}
    assert m.frame_at(row, 5, "off")["state"] == [1]
    changed = copy.deepcopy(row)
    changed["trajectory"][0]["state"] = [2]
    with pytest.raises(ValueError):
        m.frame_at(changed, 5, "off")
    with pytest.raises(ValueError):
        m.frame_at(row, 5, "on")
