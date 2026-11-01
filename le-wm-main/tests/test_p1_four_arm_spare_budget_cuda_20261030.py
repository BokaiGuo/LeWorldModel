from pathlib import Path

from experiments.spectral_reachability import p1_four_arm_spare_budget_cuda_20261030 as module


def test_protocol_and_source_closure(monkeypatch):
    monkeypatch.setattr(module.launcher.p3, "verify", lambda *args: {"sources": {}})
    monkeypatch.setattr(module.launcher, "protocol", module.protocol)
    result = module.build()
    assert result["protocol"]["node_costs"]["random"]["predict_calls"] == 84
    assert result["protocol"]["extra_indices"]["random"] == [0, 1]
    assert not result["protocol"]["fairness_proven"]
    for item in (module, module.hooks, module.control, module.bridge):
        assert str(Path(item.__file__).resolve()) in result["sources"]
    assert str(module.PROTOCOL.resolve()) in result["sources"]


def test_launcher_uses_spare_bridge(monkeypatch):
    monkeypatch.setattr(module.launcher, "NAME", module.launcher.NAME)
    monkeypatch.setattr(module.launcher, "protocol", module.launcher.protocol)
    monkeypatch.setattr(module.launcher, "build", module.launcher.build)
    monkeypatch.setattr(module.launcher.bridge, "plan_node", module.launcher.bridge.plan_node)
    module.configure()
    assert module.launcher.bridge.plan_node is module.plan_node
    assert module.launcher.NAME == module.NAME
