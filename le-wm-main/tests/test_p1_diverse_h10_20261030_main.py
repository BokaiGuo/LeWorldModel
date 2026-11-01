import copy
import hashlib
import json

import pytest

from experiments.spectral_reachability import p1_diverse_h10_20261030_main as experiment


def test_fixed_plan_has_all_slots_and_common_actions():
    path = experiment.ROOT / "results/p1_diverse_h10_20261030_main_prepare_RUN01.json"
    expected = "1beac317193419c0e1ff38e4b6b921251c5b2732f9c7e5ea4a6ae129e796c5d1"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    p = json.loads(path.read_bytes())
    old = json.loads(experiment.OLD.read_bytes())
    assert p["fixed_branches"] == 126
    assert [a["logical_pair"] for a in p["anchors"]] == [2, 6, 7]
    for anchor, bank in zip(p["anchors"], p["probe_banks"], strict=True):
        assert bank[:41] == old["probe_banks"][anchor["logical_pair"]][:41]
        assert len(anchor["trace"]) == 51
        assert anchor["prefix5_float32_hex"] != old["anchors"][3 * anchor["logical_pair"]]["prefix5_float32_hex"]


def test_verify_refuses_changed_saved_source(monkeypatch, tmp_path):
    value = {"fixed_branches": 126}
    file = tmp_path / "prepared.json"
    file.write_text(json.dumps(value))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(experiment, "build", lambda: (copy.deepcopy(value), {}))
    assert experiment.verify(str(file), digest)[0] == value
    monkeypatch.setattr(experiment, "build", lambda: ({"fixed_branches": 125}, {}))
    with pytest.raises(Exception, match="prepare drift"):
        experiment.verify(str(file), digest)
