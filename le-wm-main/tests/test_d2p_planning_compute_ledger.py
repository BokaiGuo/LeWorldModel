"""Pure accounting fixtures: no model imports or neural computation."""
from dataclasses import FrozenInstanceError, fields, replace
import json

import pytest

from experiments.spectral_reachability.d2p_planning_compute_ledger import (
    Binding, Resources, Ledger, BudgetContract, compare_pair,
)


def resources(**updates):
    # Explicit complete counters: missing instrumentation must not mean zero.
    return Resources(**{f.name: updates.get(f.name, 0) for f in fields(Resources)})


def binding(**updates):
    return replace(Binding("protocol-sha", "Reacher/two_joint", "root-rgb-sha", "weights-sha"), **updates)


def test_calls_samples_backward_horizon_and_execution_remain_separate():
    limits = resources(predict_calls=2, predict_samples=128, encoder_forwards=1,
                       action_vjp_rows=192, action_backward_calls=1,
                       max_planning_horizon_tokens=3, physical_env_steps=15)
    baseline = Ledger(binding(), "cem", limits)
    candidate = Ledger(binding(), "compiler", replace(limits, predict_calls=1,
                       predict_samples=129, action_vjp_rows=384))
    report = compare_pair(BudgetContract(binding(), limits), baseline, candidate)
    assert report["candidate_vs_budget"]["predict_calls"] == "under"
    assert report["candidate_vs_budget"]["predict_samples"] == "over"
    assert report["candidate_vs_baseline"]["action_vjp_rows"] == "over"
    assert report["candidate_vs_baseline"]["action_backward_calls"] == "equal"
    assert report["candidate_vs_baseline"]["physical_env_steps"] == "equal"
    assert report["candidate_vs_baseline"]["max_planning_horizon_tokens"] == "equal"
    assert not report["equal_resources"]
    assert not report["both_within_budget"]
    assert "total" not in json.dumps(report)


@pytest.mark.parametrize("name", [f.name for f in fields(Resources)])
@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None])
def test_every_counter_rejects_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        resources(**{name: value})


@pytest.mark.parametrize("name", [f.name for f in fields(Binding)])
@pytest.mark.parametrize("position", ["baseline", "candidate", "contract"])
def test_mixed_bindings_fail_closed(name, position):
    good, bad = binding(), binding(**{name: "different"})
    contract = BudgetContract(bad if position == "contract" else good, resources())
    baseline = Ledger(bad if position == "baseline" else good, "baseline", resources())
    candidate = Ledger(bad if position == "candidate" else good, "candidate", resources())
    with pytest.raises(ValueError, match="binding"):
        compare_pair(contract, baseline, candidate)


def test_frozen_contract_and_exact_match_are_not_an_efficacy_claim():
    contract = BudgetContract(binding(), resources())
    with pytest.raises(FrozenInstanceError):
        contract.limits.predict_calls = 2
    report = compare_pair(contract, Ledger(binding(), "a", resources()), Ledger(binding(), "b", resources()))
    assert report["both_within_budget"] and report["equal_resources"]
    assert set(report["candidate_vs_budget"].values()) == {"equal"}
    assert report["efficacy_claim"] is False


def test_missing_counters_are_not_silently_zero_filled():
    with pytest.raises(TypeError):
        Resources(predict_calls=1)


@pytest.mark.parametrize("bad", ["", " ", None, 7])
def test_empty_binding_or_method_rejected(bad):
    with pytest.raises(ValueError):
        binding(root_id=bad)
    with pytest.raises(ValueError):
        Ledger(binding(), bad, resources())


def test_equal_over_budget_is_not_compliant():
    observed = resources(encoder_forwards=13, input_vjp_rows=192,
                         input_backward_calls=192, physical_branches=19,
                         physical_action_tokens=57, physical_env_steps=285)
    report = compare_pair(BudgetContract(binding(), resources()),
                          Ledger(binding(), "a", observed), Ledger(binding(), "b", observed))
    assert report["equal_resources"] and not report["both_within_budget"]


def test_counter_types_and_duplicate_methods_fail_closed():
    with pytest.raises(TypeError):
        Ledger(binding(), "a", {})
    with pytest.raises(TypeError):
        BudgetContract(binding(), {})
    with pytest.raises(TypeError):
        Ledger({}, "a", resources())
    with pytest.raises(ValueError, match="distinct"):
        compare_pair(BudgetContract(binding(), resources()),
                     Ledger(binding(), "a", resources()), Ledger(binding(), "a", resources()))


def test_batched_horizon_fixture_does_not_confuse_trajectories_and_predict_work():
    # Four CEM iterations, 32 trajectories each, horizon 3: 12 predict
    # invocations, 384 predicted samples, 128 trajectory evaluations.
    cem = resources(predict_calls=12, predict_samples=384, rollout_samples=128,
                    imagined_action_tokens=384, max_planning_horizon_tokens=3,
                    physical_action_tokens=3, physical_env_steps=15, physical_branches=1)
    longer = replace(cem, predict_calls=16, predict_samples=512,
                     imagined_action_tokens=512, max_planning_horizon_tokens=4)
    report = compare_pair(BudgetContract(binding(), cem), Ledger(binding(), "cem", cem),
                          Ledger(binding(), "longer", longer))
    assert report["candidate_vs_baseline"]["rollout_samples"] == "equal"
    for name in ("predict_calls", "predict_samples", "imagined_action_tokens",
                 "max_planning_horizon_tokens"):
        assert report["candidate_vs_baseline"][name] == "over"
    assert report["candidate_vs_baseline"]["physical_action_tokens"] == "equal"


def test_baseline_excess_is_not_hidden_by_cheaper_candidate():
    report = compare_pair(BudgetContract(binding(), resources()),
                          Ledger(binding(), "baseline", resources(predict_calls=1)),
                          Ledger(binding(), "candidate", resources()))
    assert report["baseline_vs_budget"]["predict_calls"] == "over"
    assert report["candidate_vs_budget"]["predict_calls"] == "equal"
    assert report["candidate_vs_baseline"]["predict_calls"] == "under"
    assert not report["both_within_budget"]


def test_returned_report_is_detached_and_json_serializable():
    contract = BudgetContract(binding(), resources())
    report = compare_pair(contract, Ledger(binding(), "a", resources()), Ledger(binding(), "b", resources()))
    report["limits"]["predict_calls"] = 999
    report["binding"]["root_id"] = "changed"
    assert contract.limits.predict_calls == 0
    assert contract.binding.root_id == "root-rgb-sha"
    json.loads(json.dumps(report))
