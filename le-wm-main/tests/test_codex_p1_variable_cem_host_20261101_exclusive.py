import importlib.util
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_variable_cem_host_20261101_exclusive as m
from experiments.spectral_reachability import codex_p1_variable_cem_20261101_exclusive as variable
from experiments.spectral_reachability.codex_e1_proposal_durable_events_20261101 import ProposalJournal, read_events


def fixture_module():
    path = Path(__file__).with_name("test_codex_p1_variable_cem_20261101_exclusive.py")
    spec = importlib.util.spec_from_file_location("variable_cem_testfixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_contract_and_protocol_are_consistent():
    assert [m.node_cost(b)["predict_calls"] for b in (0, 5, 10, 15)] == [195, 194, 193, 193]
    p = m.make_protocol({"root_slot": 0, "boundaries": [0, 5, 10, 15],
                         "feedback_horizons": [3, 2, 1, 1]}, 0)
    assert p["node_costs"] == {str(b): m.node_cost(b) for b in (0, 5, 10, 15)}
    assert not p["independent_confirmation"] and not p["equal_compute"]


def test_live_controller_event_verification_and_missing_event_rejection(tmp_path):
    c, _, path = fixture_module().setup(tmp_path)
    c.step()
    m.verify_node_events(path, 0)
    c.close()
    broken = tmp_path / "bad.jsonl"
    with ProposalJournal(broken, run_id="x", arm="full_cem", proposal_kind="test") as j:
        j.append({"type": "variable_bank_contract", "contract": variable.contract()})
    with pytest.raises(ValueError):
        m.verify_node_events(broken, 0)


def test_unexpected_contract_and_bad_boundary_fail_closed(tmp_path):
    path = tmp_path / "bad.jsonl"
    with ProposalJournal(path, run_id="x", arm="full_cem", proposal_kind="test") as j:
        j.append({"type": "variable_bank_contract", "contract": {}})
    with pytest.raises(ValueError):
        m.verify_node_events(path, 0)
    with pytest.raises(ValueError):
        m.node_cost(20)


@pytest.mark.parametrize("mutation", ["missing_score", "wrong_winner", "missing_return", "wrong_rollout_boundary"])
def test_semantic_journal_corruption_rejected_even_with_valid_chain(tmp_path, mutation):
    c, _, path = fixture_module().setup(tmp_path)
    c.step()
    c.close()
    events = [r["payload"]["event"] for r in read_events(path)]
    if mutation == "missing_score":
        events.pop(next(i for i, e in enumerate(events) if e.get("type") == "score"))
    elif mutation == "wrong_winner":
        next(e for e in events if e.get("type") == "selection")["index"] = 999
    elif mutation == "missing_return":
        attempt_index = next(i for i, e in enumerate(events)
                             if e.get("operation") == "predict" and e["type"] == "attempt")
        events.pop(next(i for i, e in enumerate(events)
                        if e.get("type") == "return" and e["attempt_index"] == attempt_index))
    else:
        next(e for e in events if e.get("operation") == "rollout")["details"]["boundary"] = 5
    changed = tmp_path / "changed.jsonl"
    with ProposalJournal(changed, run_id="x", arm="full_cem", proposal_kind="test") as journal:
        for event in events:
            journal.append(event)
    with pytest.raises(ValueError):
        m.verify_node_events(changed, 0)


def test_legacy_prepares_and_cross_root_names_rejected():
    with pytest.raises(ValueError):
        m.check_prepare_name("codex_p1_full_cem_host_v2_20261101_exclusive_root0_PREP_MAIN01.json", 0)
    with pytest.raises(ValueError):
        m.check_prepare_name(m.NAME + "_root0_PREP_A.json", 1)


def test_real_durable_driver_and_analytic_leaf_cpu_fixture(tmp_path):
    import json
    import torch
    from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
    from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
    from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer

    task_path = (Path(__file__).resolve().parents[1] / "results" /
                 "p1_first_feedback_carry_20261101_main_root0_PREP_MAIN02.json")
    task = json.loads(task_path.read_bytes())["source"]["tasks"][0]
    contract = m.node_contract(task, "a"*64, "b"*64, 0)
    digest = m.old.spec.digest(contract)
    driver = durable.Driver(tmp_path/"ledger", contract, digest, "native", "node", "fixture")
    class Model:
        def encode(self, info):
            return info["pixels"]
        def action_encoder(self, actions):
            return actions
        def predict(self, states, actions):
            return states + actions[:, -1, :].mean(dim=-1, keepdim=True)[:, None, :]
    leaf = leafmod.LeafAdapter(Model(), driver, contract, digest, analytic_fixture=True)
    goal, matrix = torch.ones(1, 2), torch.eye(2)
    mean, std = torch.zeros(2), torch.ones(2)
    identity = {"model_sha256": "a"*64, "root_sha256": "b"*64, "task_sha256": "c"*64,
                "goal_sha256": variable.base.tensor_sha(goal), "metric_sha256": variable.base.tensor_sha(matrix),
                "normalizer_sha256": variable.base.tensor_sha(torch.stack((mean, std)))}
    def observe(boundary, meter):
        return {"identity": identity, "rgb_sha256": "d"*64, "provenance": "fixture",
                "encoding_mode": "inherited", "states": torch.zeros(1, 3, 2),
                "history": torch.zeros(1, 2, 10), "goal": goal, "matrix": matrix, "mean": mean, "std": std}
    path = tmp_path/"cem.jsonl"
    def issue(packet):
        ledger = driver.finish()
        account = consumer.consume(ledger["journal"], ledger["sha256"], digest, "native", "node", "fixture")
        assert account["charged_upper_bounds"] == m.node_cost(0)
        m.verify_node_events(path, 0)
        return {"executed_steps": 1, "stopped": True}
    controller = variable.Controller(leaf=leaf, observe=observe, issue=issue, journal=path,
                                     run_id="fixture", identity=identity, seed=m.old.SEED)
    assert controller.step()["receipt"]["stopped"]
    controller.close()


@pytest.mark.parametrize("mode", ["full", "early_stop", "bad_account", "bad_events"])
def test_isolated_host_orchestration_fail_closed(tmp_path, monkeypatch, mode):
    path = Path(__file__).with_name("test_codex_p1_full_cem_host_20261101_exclusive.py")
    loader = importlib.util.spec_from_file_location("host_fixture", path)
    fixture = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(fixture)
    prepared, data, store, runtime, factory, runner, rebind = fixture.fixture_runtime(
        tmp_path, monkeypatch, stop_step=7 if mode == "early_stop" else None)
    monkeypatch.setattr(m, "node_contract", lambda task, model, hardware, b:
                        {"boundary": b, "caps": m.node_cost(b)})
    if mode == "bad_account":
        original = m.old.consumer.consume
        def bad_account(*args):
            account = original(*args)
            account["charged_upper_bounds"]["predict_calls"] -= 1
            return account
        monkeypatch.setattr(m.old.consumer, "consume", bad_account)
    if mode == "bad_events":
        monkeypatch.setattr(m, "verify_node_events",
                            lambda *args: (_ for _ in ()).throw(ValueError("bad event")))
    execute = m.isolated_execute()
    result = execute(prepared, data, store, None, runtime, factory, runner, rebind, 0)
    dispatched = [event for event in store.events if event[0] == "authorized_dispatch"]
    if mode == "full":
        assert result["status"] == "recorded_known_root_not_admitted"
        assert result["rows"][0]["control_returned"] == len(dispatched) == 20
        assert [n["status"] for n in result["nodes"]["s0_full_cem"]] == ["sealed"] * 4
        assert [n["receipt"]["account"]["charged_upper_bounds"]["predict_calls"]
                for n in result["nodes"]["s0_full_cem"]] == [195, 194, 193, 193]
    elif mode == "early_stop":
        assert result["status"] == "recorded_known_root_not_admitted"
        assert result["rows"][0]["control_returned"] == len(dispatched) == 7
        assert [n["status"] for n in result["nodes"]["s0_full_cem"]] == [
            "sealed", "sealed", "not_started", "not_started"]
    else:
        assert result["status"] == "partial_or_failed"
        assert len(dispatched) == 0
