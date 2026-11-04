import json

import numpy as np
import pytest

from experiments.spectral_reachability import pc2_auto_response_unicycle_dev_20261103_codex as pc


def test_bank_response_and_legal_actions():
    goal = np.array([0.3, 0.8])
    bank, records = pc.build_bank(goal, 0, "goal")
    assert len(bank) == 5 and len(records) == 4
    for item, record in zip(bank[1:], records):
        assert np.all(np.isfinite(record["jacobian"]))
        assert abs(item["prefix"][0]) <= 1
        assert abs(item["prefix"][1]) <= pc.dynamics.TURN
        np.testing.assert_allclose(
            item["prefix"], pc.prefix_from_normalized(record["candidate"])
        )
    with pytest.raises(ValueError):
        pc.prefix_from_normalized([1.1, 0])


def test_seal_and_intervention_bind_actual_selected_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "PLANS", tmp_path / "plans.json")
    monkeypatch.setattr(pc, "INTERVENTION", tmp_path / "intervention.json")
    pc.plan_stage()
    plans = json.loads(pc.PLANS.read_text())
    with pytest.raises(FileExistsError):
        pc.plan_stage()
    pc.intervention_stage()
    intervention = json.loads(pc.INTERVENTION.read_text())
    assert len(plans["rows"]) == len(intervention["rows"]) == 16
    for plan, result in zip(plans["rows"], intervention["rows"]):
        for name, method in plan["methods"].items():
            bank = plan["random_bank"] if name == "random_bridge" else plan["response_bank"]
            selected = bank[method["selected_index"]]
            assert method["chain"] == [selected["prefix"], selected["suffix"]]
            observed = result["methods"][name]
            assert observed["chain"] == method["chain"]
            controls = observed["controls"]
            assert controls["selected"]["prefix"] == method["chain"][0]
            for control in controls.values():
                assert len(control["common_suffix_distances"]) == 6
                assert control["active_changes"][2] == 0
                assert control["own_optimized_distance"] <= min(control["common_suffix_distances"][:5]) + 1e-12
            assert observed["limited_witness"] == (
                observed["nonprogress"] and observed["active_response"] and observed["own_plan_advantage"]
            )
    with pytest.raises(FileExistsError):
        pc.intervention_stage()
