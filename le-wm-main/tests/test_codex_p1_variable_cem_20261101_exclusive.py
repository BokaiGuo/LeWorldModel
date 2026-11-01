import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_p1_variable_cem_20261101_exclusive as m
from experiments.spectral_reachability.codex_e1_proposal_durable_events_20261101 import read_events


def test_common_cap_and_replay_reserve():
    contract = m.contract()
    assert contract["samples"] == [16, 24, 48, 48]
    assert contract["forward_per_node"] == [195, 194, 193, 193]
    assert contract["common_forward_cap"] == 195
    assert contract["claim"] == "development_only_not_confirmatory"
    assert contract["structured_b15"] == "unknown_not_started"
    assert not contract["equal_compute"]


class Leaf:
    def action_encode(self, actions, *, phase):
        return actions

    def predict(self, states, actions, *, phase):
        return states + actions[:, -1, :].mean(dim=-1, keepdim=True)[:, None, :]


def setup(tmp_path, stop=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    goal, matrix = torch.ones(1, 2), torch.eye(2)
    mean, std = torch.zeros(2), torch.ones(2)
    identity = dict(model_sha256="a"*64, root_sha256="b"*64, task_sha256="c"*64,
                    goal_sha256=m.base.tensor_sha(goal), metric_sha256=m.base.tensor_sha(matrix),
                    normalizer_sha256=m.base.tensor_sha(torch.stack((mean, std))))
    def observe(boundary, leaf):
        return dict(identity=identity, rgb_sha256="d"*64, provenance="fixture",
                    encoding_mode="inherited", states=torch.zeros(1, 3, 2),
                    history=torch.zeros(1, 2, 10), goal=goal, matrix=matrix, mean=mean, std=std)
    packets = []
    def issue(packet):
        packets.append(packet)
        return {"executed_steps": 1 if stop else 5, "stopped": stop}
    c = m.Controller(leaf=Leaf(), observe=observe, issue=issue,
                     journal=tmp_path/"cem.jsonl", run_id="fixture", identity=identity, seed=11)
    return c, packets, tmp_path/"cem.jsonl"


def test_controller_runs_variable_banks_and_binds_contract(tmp_path):
    c, packets, path = setup(tmp_path)
    results = [c.step() for _ in range(4)]
    c.close()
    assert len(packets) == 4
    assert [r["horizon"] for r in results] == [3, 2, 1, 1]
    events = [e["payload"]["event"] for e in read_events(path)]
    assert events[1]["type"] == "variable_bank_contract"
    assert events[1]["contract"] == m.contract()
    for b, samples, calls in zip((0, 5, 10, 15), (16, 24, 48, 48), (195, 194, 193, 193)):
        banks = [e for e in events if e["type"] == "bank" and e["boundary"] == b]
        assert len(banks) == 4 and all(len(e["raw"]) == samples for e in banks)
        scores = [e for e in events if e["type"] == "score" and e["boundary"] == b]
        assert len(scores) == 4*samples
        assert (len(scores) + 1) * dict(zip((0, 5, 10, 15), (3, 2, 1, 1)))[b] == calls


def test_partial_stop_and_failure_never_dispatch_next_packet(tmp_path):
    c, packets, _ = setup(tmp_path, stop=True)
    assert c.step()["receipt"]["executed_steps"] == 1
    with pytest.raises(RuntimeError):
        c.step()
    assert len(packets) == 1
    c.close()


def test_reject_invalid_cap_and_boolean():
    with pytest.raises(ValueError):
        m.allocate(True, 3)
    with pytest.raises(ValueError):
        m.allocate(50, 3)


def test_controller_instances_do_not_change_old_globals(tmp_path):
    import copy
    a, _, _ = setup(tmp_path / "a")
    b, _, _ = setup(tmp_path / "b")
    a.step()
    b.step()
    a.step()
    assert m.base.SAMPLES == 16
    assert b._step_globals["SAMPLES"] == 16
    assert a._step_globals["SAMPLES"] == 24
    assert copy.deepcopy(m.contract())["samples"] == [16, 24, 48, 48]
    a.close()
    b.close()


def test_node_cost_is_explicit_and_out_of_range_rejected():
    for boundary, value in zip((0, 5, 10, 15), (195, 194, 193, 193)):
        cost = m.node_cost(boundary)
        assert cost["predict_calls"] == cost["predict_samples"] == cost["action_encode_calls"] == value
        assert cost["backward_calls"] == 0
    with pytest.raises(ValueError):
        m.node_cost(20)


def test_development_source_hashes_and_real_observed_node_accounting():
    result = m.audit_development_sources()
    assert [row["forward"] for row in result] == [[174, 192, 183], [174, 192, 183]]
    assert all(row["unstarted_boundaries"] == [15] for row in result)
    assert all(not row["independent_confirmation"] for row in result)


def test_real_node_contract_binds_variable_budget_and_optimizer_source():
    import json
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "results" /
            "p1_first_feedback_carry_20261101_main_root0_PREP_MAIN02.json")
    task = json.loads(path.read_bytes())["source"]["tasks"][0]
    for boundary in (0, 5, 10, 15):
        value = m.node_contract(task, "a"*64, "b"*64, boundary, seed=20261101)
        assert value["levels"][0]["caps"] == m.node_cost(boundary)
        assert value["identity"]["optimizer"] == m.source_sha()
        assert value["identity"]["optimizer_config"] == m.spec.digest(m.node_config(boundary, 20261101))
