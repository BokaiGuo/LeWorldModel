import numpy as np
import pytest
import json

from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as exp


@pytest.mark.parametrize("arm", ["goal", "cem"])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_fixed_terminal_budget_carry_and_candidate_binding(arm, step):
    meter = exp.base.Meter(exp.CAP)
    def predict(chains):
        def raw(batch):
            positions = batch.sum(axis=1)
            values = np.square(positions - [.3, -.2]).sum(axis=1)
            return positions, values
        return meter.predict(np.asarray(chains), raw)
    carry = np.zeros((15 - 5 * step, 2)) if step else None
    result = exp.decide(arm, 0, step, np.array([.3, -.2]), predict, meter, carry)
    assert len(result["selected"]["chain"]) == 15 - 5 * step
    assert meter.token_samples <= exp.CAP
    assert exp.CAP - meter.token_samples < (15 - 5 * step) // 5
    if carry is not None:
        assert result["carry"] == carry.tolist()
        assert result["selected"]["terminal_score"] <= result["carry_score"] + 1e-12
    if arm == "goal":
        assert result["spectral_counts"]["svd"] == 1
        assert any(item["name"] == "zero" for item in result["programs"])
    else:
        assert result["spectral_counts"] == {}


def test_missing_or_wrong_carry_fails_closed():
    with pytest.raises(ValueError):
        exp.decide("cem", 0, 1, np.zeros(2), None, exp.base.Meter(exp.CAP))
    with pytest.raises(ValueError):
        exp.decide("goal", 0, 1, np.zeros(2), None, exp.base.Meter(exp.CAP), np.zeros((5, 2)))


def test_saved_real_feedback_action_clock_and_physical_receipts():
    physical_path = exp.PREFIX.with_name(exp.PREFIX.name + "_physical.json")
    if not physical_path.exists():
        pytest.skip("real feedback panel not run")
    physical = json.loads(physical_path.read_text())
    assert len(physical["rows"]) == 8
    assert physical["backend"] == "2.1.2.14"
    for row in physical["rows"]:
        decisions, execution, images = exp.paths(row["root"], row["arm"])
        receipt = json.loads(execution.read_text())
        assert row["execution_sha256"] == exp.old.prior.sha(execution)
        assert row["images_sha256"] == exp.old.prior.sha(images)
        previous = None
        actions = []
        for step, (path, digest) in enumerate(zip(decisions, row["decision_sha256"], strict=True)):
            assert exp.old.prior.sha(path) == digest
            d = json.loads(path.read_text())
            chain = d["result"]["selected"]["chain"]
            assert d["result"]["horizon"] == len(chain) == 15 - 5 * step
            assert d["counts"]["prediction_token_samples"] <= exp.CAP
            assert receipt["rows"][step]["actions"] == chain[:5]
            if previous is not None:
                assert d["result"]["carry"] == previous[5:]
                assert d["result"]["selected"]["terminal_score"] <= d["result"]["carry_score"] + 1e-12
            previous = chain
            actions.extend(chain[:5])
        assert actions == receipt["executed_actions"]
        assert len(row["trace"]) == 31
        distances = np.linalg.norm(np.asarray([x["state"][:2] for x in row["trace"]]) - [1.8, 2.8], axis=1)
        np.testing.assert_allclose(distances[[15, 20, 25, 30]], row["distances"], atol=1e-12)
