import pytest

from experiments.spectral_reachability import p1_four_arm_spare_budget_bridge_20261030 as bridge
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from test_codex_four_arm_common_suffix_bridge_20261030 import Store
from test_codex_four_arm_common_suffix_control_20261030 import task
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


@pytest.fixture
def fixture_runtime(monkeypatch):
    original = leafmod.LeafAdapter.__init__
    monkeypatch.setattr(leafmod.LeafAdapter, "__init__",
                        lambda self, model, driver, contract, contract_sha:
                        original(self, model, driver, contract, contract_sha, analytic_fixture=True))
    return {"model": Affine(), "model_sha256": "c" * 64, "hardware_sha256": "d" * 64,
            "device": "cpu", "check_limits": lambda: None}


@pytest.mark.parametrize("arm", ("structured", "native", "random", "immediate_progress"))
def test_bridge_persists_and_consumes_complete_node(tmp_path, fixture_runtime, arm):
    store = Store(tmp_path)
    result = bridge.plan_node(task(), arm, 0, store, fixture_runtime)
    assert result["account"]["charged_upper_bounds"] == bridge.control.cost(arm)
    assert len(result["commands"]) == 5
    assert (tmp_path / f"s0_{arm}_b00_spare" / "candidates.jsonl").is_file()


def test_limit_failure_never_returns_commands(tmp_path, fixture_runtime):
    def fail():
        raise RuntimeError("cap")
    fixture_runtime["check_limits"] = fail
    with pytest.raises(ValueError, match="node failed"):
        bridge.plan_node(task(), "random", 0, Store(tmp_path), fixture_runtime)
