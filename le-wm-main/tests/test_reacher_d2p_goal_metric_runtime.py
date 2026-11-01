"""Pure numeric score fixtures; neural paths must reject CPU before inference."""
import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
from experiments.spectral_reachability.reacher_d2p_goal_metric_runtime import measure_goal, score_bank, MODES, declared_encoder_source
from experiments.spectral_reachability.bounded_response_qp import certified_response_gain


def fixture():
    response = torch.eye(2, dtype=torch.float64)
    residual = torch.tensor([.3, -.2], dtype=torch.float64)
    gain = certified_response_gain(response, residual, damping=.01)
    bank = [{"index": i, "prefix_cost": .2+i/100., "terminal_cost": .3+i/100.,
        "prefix_vector": [.1+i/100., -.1], "terminal_vector": [.2, -.2+i/100.],
        "response": response.tolist(), "common_response_gain": gain} for i in range(17)]
    return {"bank": bank, "common_residual": residual.tolist()}


def test_neural_cpu_fallback_rejected_without_calling_model():
    with pytest.raises(ValueError, match="CPU fallback forbidden"):
        measure_goal(object(), torch.zeros(1,1,3,224,224), [], seed=1)


def test_encoder_source_follows_checkpoint_class_without_instantiation(tmp_path, monkeypatch):
    import hydra.utils
    class Declared:
        def __new__(cls): raise AssertionError("must not instantiate model on CPU")
        def encode(self, info): return info
    def resolve(name):
        assert name == "declared.package.Class"
        return Declared
    monkeypatch.setattr(hydra.utils,"get_class",resolve)
    (tmp_path/"config.json").write_text(json.dumps({"_target_":"declared.package.Class"}))
    assert declared_encoder_source(tmp_path) == Path(__file__)


def test_both_modes_preserve_original_bank_and_own_costs():
    plan = fixture()
    saved = copy.deepcopy(plan)
    metric = GoalEncoderMetric(torch.diag(torch.tensor([.01, .04], dtype=torch.float64)))
    target = torch.tensor([.4,-.3], dtype=torch.float64)
    result = score_bank(metric, plan, target)
    assert plan == saved
    assert set(result) == set(MODES)
    for old, unchanged, weighted in zip(plan["bank"], result[MODES[0]]["bank"], result[MODES[1]]["bank"]):
        assert unchanged["prefix_cost"] == old["prefix_cost"]
        assert unchanged["terminal_cost"] == old["terminal_cost"]
        assert unchanged["common_response_gain"] == weighted["common_response_gain"]
        expected = metric.apply(torch.tensor(old["terminal_vector"],dtype=torch.float64)-target).square().mean()
        assert weighted["terminal_cost"] == float(expected)
    assert all(result[m]["decision"]["physical_labels_used"] is False for m in MODES)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "reordered"])
def test_incomplete_bank_fails_closed(defect):
    plan = fixture()
    if defect == "missing": plan["bank"].pop()
    if defect == "duplicate": plan["bank"][1]["index"] = 0
    if defect == "reordered": plan["bank"].reverse()
    with pytest.raises(ValueError, match="complete ordered"):
        score_bank(GoalEncoderMetric(torch.eye(2,dtype=torch.float64)),plan,[0.,0.])
