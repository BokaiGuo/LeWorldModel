import pytest

from experiments.pc_goal_mode_rgb_physical_20261103_codex import validate


def test_selected_chain_and_receipt_binding():
    selected = {"name": "x", "prefix": [[0., 0.]] * 5, "suffix": [[.1, 0.]] * 10}
    chain = selected["prefix"] + selected["suffix"]
    plan = {"result": {"selected": selected, "programs": [selected]}, "chain": chain}
    execution = {"plan_sha256": "plan", "images_sha256": "image", "chain": chain}
    assert validate(plan, execution, "plan", "image")["x"] == chain
    with pytest.raises(ValueError):
        validate(plan, execution, "different", "image")
    execution["chain"] = [[0., 0.]] * 15
    with pytest.raises(ValueError):
        validate(plan, execution, "plan", "image")
