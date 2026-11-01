import copy
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_rollout_contract_20261030_codex_v1 as contract


@pytest.fixture(scope="module")
def prepared():
    return contract.prepare(*contract.load_sources())


def test_complete_panel_and_no_neural_import(prepared):
    assert len(prepared["rows"]) == 164
    assert len(prepared["pairs"]) == 80
    assert prepared["planned_calls"] == {"encode_state": 164, "predict_next": 820}
    assert not any(prepared["actual_calls"].values())
    assert "torch" not in sys.modules


def test_actual_input_abi(prepared):
    row = prepared["rows"][0]
    values = contract.materialize(row)
    assert values["actions"].shape == (1, 5, 2)
    assert values["image"].shape == (1, 3, 64, 64)
    contract.validate_input(row, values)


@pytest.mark.parametrize("damage", ["action", "image", "goal", "shape", "dtype", "nan"])
def test_reject_input_mutations(prepared, damage):
    row = prepared["rows"][0]
    values = contract.materialize(row)
    if damage == "action":
        values["actions"][0, 0, 0] += 1
    elif damage == "image":
        values["image"][0, 0, 0, 0] += 0.1
    elif damage == "goal":
        values["goal"] = np.zeros(1)
    elif damage == "shape":
        values["actions"] = values["actions"][:, :4]
    elif damage == "dtype":
        values["actions"] = values["actions"].astype(np.float64)
    else:
        values["actions"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        contract.validate_input(row, values)


def test_reject_missing_branch():
    previous, physical, panel = contract.load_sources()
    previous["rows"].pop()
    with pytest.raises(ValueError, match="row counts"):
        contract.prepare(previous, physical, panel)


def test_reject_pair_schedule():
    previous, physical, panel = contract.load_sources()
    panel["pairs"][0]["coord"] = 1
    previous["pairs"] = copy.deepcopy(panel["pairs"])
    with pytest.raises(ValueError, match="pair schedule"):
        contract.prepare(previous, physical, panel)
