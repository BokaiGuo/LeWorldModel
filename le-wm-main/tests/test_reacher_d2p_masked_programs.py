"""CPU synthetic logic only, not learned-model experimental evidence."""
import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_programs as programs
from experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot import check_raw, physical_row


class CausalFixture:
    def action_encoder(self, values):
        return values
    def predict(self, emb, actions):
        # Uses all history so the test detects a lost/duplicated prefix action.
        offsets = actions[..., :2] + .2*actions[..., 2:4]
        offsets = offsets + .1*offsets.roll(1, dims=1)
        return emb + offsets + .03*emb.roll(1, dims=1)


@pytest.mark.parametrize("task", chart.TASKS)
def test_imagined_prefix_history_exactly_matches_full_action_sequence(task):
    model = CausalFixture()
    emb = torch.arange(6.).reshape(1, 3, 2)/10
    history = torch.arange(30.).reshape(1, 3, 10)/30
    mean,std = torch.linspace(-.1,.2,10),torch.linspace(.5,1,10)
    width = len(chart.active_columns(task))
    commands = torch.linspace(-.75,.75,3*width).reshape(1,3,width)
    after,actions,_ = programs.after_prefix(model,emb,history,commands[:,:1],task=task,mean=mean,std=std)
    split = chart.terminal(model,after,actions,commands[:,1:],task=task,mean=mean,std=std)
    full = chart.terminal(model,emb,history,commands,task=task,mean=mean,std=std)
    torch.testing.assert_close(split,full,rtol=0,atol=0)
    torch.testing.assert_close(actions[:,:2],history[:,1:])


def test_fixed_spectral_bank_contains_signed_bounded_directions_and_literal_zero():
    bank = programs.spectral_bank(torch.eye(10))
    assert bank.shape == (17,10) and torch.count_nonzero(bank[0]) == 0
    for i in range(1,17,2):
        torch.testing.assert_close(bank[i],-bank[i+1])
    assert bank.abs().max() == .75


def test_box_solver_keeps_initial_and_never_increases_model_cost():
    target = torch.tensor([3.,-3.])
    # A computed output respects no_grad like the actual frozen predictor;
    # returning the input object itself would retain its requires_grad flag.
    commands,receipt = programs.fit(lambda x:x+0,torch.zeros(2),target)
    torch.testing.assert_close(commands,torch.tensor([1.,-1.]))
    assert len(receipt["trace"]) == 4
    assert all(s["cost_after"] <= s["cost_before"] for s in receipt["trace"])
    assert receipt["physical_labels_used"] is False


def bank():
    return [{"index":i,"prefix_cost":1. if i != 1 else .5,"terminal_cost":1.,
        "common_response_gain":{"bounded_quadratic_gain":.1,"bounded_quadratic_gain_upper_bound":.1,
            "objective_certificate_passed":True,"physical_labels_used":False}} for i in range(17)]


def test_selector_requires_non_greedy_and_future_cost_and_certified_geometry():
    rows = bank()
    assert programs.select(rows)["native_index"] == 1
    rows[2]["prefix_cost"],rows[2]["terminal_cost"] = 1.1,.5
    assert not programs.select(rows)["reconfiguration_witness_found"]
    rows[2]["common_response_gain"].update(bounded_quadratic_gain=.3,bounded_quadratic_gain_upper_bound=.3)
    result = programs.select(rows)
    assert result["native_index"] == 2 and result["qualified_indices"] == [2]
    assert result["physical_labels_used"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "oracle", "uncertified", "interval"])
def test_bad_selector_evidence_is_rejected(mutation):
    rows = copy.deepcopy(bank())
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[1]["index"] = 0
    elif mutation == "nan": rows[0]["terminal_cost"] = float("nan")
    elif mutation == "oracle": rows[0]["common_response_gain"]["physical_labels_used"] = True
    elif mutation == "uncertified": rows[0]["common_response_gain"]["objective_certificate_passed"] = False
    else: rows[0]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] = 0.
    with pytest.raises(ValueError): programs.select(rows)


@pytest.mark.parametrize("task", chart.TASKS)
def test_all_physical_torques_respect_literal_mask(task):
    raw = np.zeros((15,2),np.float32)
    raw[:,list(chart.TASKS[task])] = .75
    assert check_raw(raw,task)
    raw[0,0] = 1.1
    assert not check_raw(raw,task)
    raw[:] = .1
    assert check_raw(raw,task) == (task == "full_drive")


def test_physical_result_writer_uses_its_own_sealed_root_not_last_planned_root():
    records = [{"seed":i,"source_episode":300+i,"source_start_row":1000*i,"tasks":{"unused":"planning"}} for i in range(8)]
    observed = {"full_drive":{"packets":[1,2,3]}}
    rows = [physical_row(record,observed) for record in records]
    assert [r["seed"] for r in rows] == list(range(8))
    assert len({r["source_episode"] for r in rows}) == 8
    assert rows[0]["source_start_row"] == 0 and rows[-1]["source_start_row"] == 7000
    assert all(r["tasks"] == observed for r in rows)
