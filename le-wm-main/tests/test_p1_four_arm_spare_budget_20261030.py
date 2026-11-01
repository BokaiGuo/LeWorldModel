import pytest

from experiments.spectral_reachability import p1_four_arm_spare_budget_20261030 as module
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from test_codex_four_arm_common_suffix_control_20261030 import task
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


@pytest.mark.parametrize("arm", spec.ARMS)
def test_node_accounts_for_extra_optimization(tmp_path, arm):
    example = task()
    policy, contract, config = module.contract_for(example, "c" * 64, "d" * 64)
    digest = spec.digest(contract)
    driver = durable.Driver(tmp_path / "ledger", contract, digest, arm, "node", arm)
    leaf = leafmod.LeafAdapter(Affine(), driver, contract, digest, analytic_fixture=True)
    events = []
    result = module.plan(leaf, arm, example, config, policy, contract,
                         device="cpu", emit=lambda kind, payload: events.append((kind, payload)))
    receipt = driver.finish()
    account = consumer.consume(receipt["journal"], receipt["sha256"], digest, arm, "node", arm)
    assert account["charged_upper_bounds"] == module.cost(arm)
    assert events[0][1]["extra_indices"] == list(module.EXTRA_INDICES[arm])
    assert result["optimized_raw"].shape == (8, 3, 5, 2)
    assert not result["fairness_proven"]
    if arm in ("native", "random"):
        assert module.cost(arm)["predict_calls"] == 84
        assert module.cost(arm)["backward_calls"] == 20
    assert all(module.cost(arm)[axis] <= contract["levels"][0]["caps"][axis] for axis in spec.AXES)


def test_insufficient_reserve_fails_before_any_model_work(tmp_path):
    example = task()
    policy, contract, config = module.contract_for(example, "c" * 64, "d" * 64)
    digest = spec.digest(contract)
    driver = durable.Driver(tmp_path / "ledger", contract, digest, "random", "node", "random")
    leaf = leafmod.LeafAdapter(Affine(), driver, contract, digest, analytic_fixture=True)
    driver.charged["predict_calls"] = driver.caps["predict_calls"]
    with pytest.raises(ValueError, match="full node reserve unavailable"):
        module.plan(leaf, "random", example, config, policy, contract,
                    device="cpu", emit=lambda *_: None)
    assert driver.failed
