import copy
import hashlib
import json

import pytest

from experiments.spectral_reachability import codex_p1_cp5_physical_24_20261030_v1 as m


@pytest.fixture(scope="module")
def sources():
    prepared = json.loads(m.h10.INPUT.read_bytes())
    blind = json.loads(m.plans.run.BUNDLE.read_bytes())
    receipt = json.loads(m.RECEIPT.read_bytes())
    events = [json.loads((m.PLAN_DIR/f"row_{i:02d}_result.json").read_bytes()) for i in range(24)]
    return prepared["anchors"], blind["rows"], events, receipt


def test_fixed_real_roster_and_action_identity(sources):
    roster = m.make_roster(*sources)
    assert len(roster) == 24
    assert [x["role"] for x in roster] == ["selected", "zero", "greedy"]*8
    for row in roster:
        raw = bytes.fromhex(row["raw10_float32_hex"])
        assert len(raw) == 80 and hashlib.sha256(raw).hexdigest() == row["raw10_sha256"]


@pytest.mark.parametrize("change", ["cp5", "prefix", "order", "actions"])
def test_rejects_tampered_input_before_physics(sources, change):
    anchors, blind, events, receipt = copy.deepcopy(sources)
    if change == "cp5":
        blind[0]["cp5_pixel_sha256"] = "0"*64
    elif change == "prefix":
        blind[0]["prefix_raw_f32_hex"] = "0"*80
    elif change == "order":
        events[0]["index"] = 1
    else:
        events[0]["result"]["actions"] = [[[[2, 0] for _ in range(5)] for _ in range(2)]]
    with pytest.raises(ValueError):
        m.make_roster(anchors, blind, events, receipt)


def test_single_row_caps():
    assert m.CAPS["public_step"] == 25
    assert m.CAPS["physics_step"] == 254
    assert m.CAPS["explicit_render"] == 81
