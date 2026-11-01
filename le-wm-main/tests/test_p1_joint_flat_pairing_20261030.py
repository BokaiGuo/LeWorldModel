import pytest

from experiments.spectral_reachability import p1_joint_complete_pairing_20261030 as pairing
from experiments.spectral_reachability import p1_joint_flat_pairing_20261030 as flat
from test_p1_joint_complete_pairing_20261030 import fixed_panel


def test_flat_pairing_preserves_censoring_and_rejects_bad_count():
    prepared, anchors, panel = fixed_panel()
    panel["rows"][0].update(endpoint=None, status="early_stop", endpoint_minus_own_zero=None)
    joined = pairing.join(prepared, anchors, panel)
    shards = {}
    for anchor in anchors["rows"]:
        shards[anchor["index"]] = {
            probe["name"]: {"status": "probe_completed", "own_returned": 15}
            for probe in prepared["probe_banks"][anchor["logical_pair"]]
        }
    shards[0][prepared["probe_banks"][0][0]["name"]] = {"status": "early_stop", "own_returned": 12}
    rows = flat.flatten(joined, shards)
    assert len(rows) == 1008
    assert rows[0]["common_returned_suffix_steps"] == 7
    assert rows[0]["common_endpoint_object_position_error"] is None
    assert rows[0]["prefix_native_success_before"] is False
    shards[0][prepared["probe_banks"][0][1]["name"]]["own_returned"] = 14
    with pytest.raises(ValueError, match="completed suffix length"):
        flat.flatten(joined, shards)
