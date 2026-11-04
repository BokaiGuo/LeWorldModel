from types import SimpleNamespace
import json
from collections import Counter

import numpy as np
import pytest
import torch

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as p


def test_cem_projected_elites_budget_and_tie_rule():
    prior = torch.zeros(10)
    lower, upper = torch.full((10,), -.1), torch.full((10,), .1)
    noise = torch.ones(3,16,10) * 10
    seen = []
    def evaluate(x):
        seen.append(x.clone())
        return torch.zeros(len(x))
    chosen, trace, final, scores, winner = p.cem(evaluate, prior, lower, upper, noise)
    assert [len(x) for x in seen] == [16,16,16,2]
    assert all(torch.all((x >= lower) & (x <= upper)) for x in seen)
    assert winner == 0 and torch.equal(chosen, prior)
    for candidates, _, elite, mean, std in trace:
        assert elite.tolist() == [0,1,2,3]
        torch.testing.assert_close(mean, candidates[elite].mean(0))
        torch.testing.assert_close(std, candidates[elite].std(0,correction=1).clamp_min(1e-5))
    with pytest.raises(ValueError):
        p.cem(evaluate, prior, lower, upper, noise[:2])


@pytest.mark.parametrize("n", [2,16])
def test_history_binding_actual_batch_sizes(n):
    class Predictor:
        def __init__(self):
            self.actions = []
        def encode_act(self, x):
            self.actions.append(x.clone())
            return x
        def forward_pred(self, visual, action, proprio):
            return visual, None, proprio
    predictor = Predictor()
    model = SimpleNamespace(proprio_mode="predict_proprio", model=predictor)
    ctx = {"visual":torch.zeros(1,3,1,1,1,1), "proprio":torch.zeros(1,3,1,1)}
    history = torch.arange(20.).reshape(2,1,10).expand(-1,n,-1)
    future = (torch.arange(n*10.).reshape(1,n,10)+100)
    p.unroll_with_action_history(model,ctx,history,future,3)
    expected = torch.cat((history,future)).transpose(0,1)
    assert torch.equal(predictor.actions[0], expected)


def test_roster_and_source_pin():
    source, _, names = p.files()
    assert len(names) == len(set(names)) == 32
    assert names[0] == "r8343_a0_p2" and names[-1] == "r8351_a1_p5"


def test_saved_decisions_and_optimizer_traces():
    if not p.DECISIONS.exists():
        pytest.skip("decisions not produced")
    ctx, decision = p.load_sealed(p.CONTEXT), p.load_sealed(p.DECISIONS)
    assert decision["context_sha256"] == p.sha(p.CONTEXT)
    assert decision["counts"] == {"one":1600,"three":1600,"parity_samples":36,"encodes":97,"batches":256}
    assert decision["model_before"] == decision["model_after"]
    provenance = json.loads(p.ARCHIVE.read_text())["provenance"]
    mean = torch.tensor(provenance["action_mean"])
    std = torch.tensor(provenance["action_std"])
    lower, upper = ((-1-mean)/std).repeat(5), ((1-mean)/std).repeat(5)
    prior = ((-mean)/std).repeat(5)
    with np.load(p.CONTEXT.with_suffix(".npz"), allow_pickle=False) as context, \
         np.load(p.DECISIONS.with_suffix(".npz"), allow_pickle=False) as arrays:
        for i, row in enumerate(decision["rows"]):
            name = row["name"]
            assert row["seed"] == 2026110300+i
            expected_noise = torch.randn(3,16,10,generator=torch.Generator().manual_seed(row["seed"]))
            torch.testing.assert_close(torch.from_numpy(arrays[name+"_noise"]), expected_noise, rtol=0, atol=0)
            history = ((torch.from_numpy(context[name+"_actions"][2:12])-mean)/std).reshape(2,1,10)
            torch.testing.assert_close(torch.from_numpy(arrays[name+"_history_packets"]), history)
            for arm in ("one","three"):
                m, s = prior.clone(),torch.full_like(prior,1/3)
                for j in range(3):
                    candidate = torch.from_numpy(arrays[f"{name}_{arm}_{j}_candidates"])
                    expected = (m+expected_noise[j]*s).clamp(lower,upper)
                    expected[0] = m.clamp(lower,upper)
                    torch.testing.assert_close(candidate, expected, rtol=1e-5, atol=1e-6)
                    scores = torch.from_numpy(arrays[f"{name}_{arm}_{j}_scores"])
                    elite = torch.from_numpy(arrays[f"{name}_{arm}_{j}_elite"])
                    assert torch.equal(elite,torch.argsort(scores,stable=True)[:4])
                    m, s = candidate[elite].mean(0),candidate[elite].std(0,correction=1).clamp_min(1e-5)
                    torch.testing.assert_close(torch.from_numpy(arrays[f"{name}_{arm}_{j}_mean"]),m,rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(torch.from_numpy(arrays[f"{name}_{arm}_{j}_std"]),s,rtol=1e-5,atol=1e-6)
                final = torch.from_numpy(arrays[f"{name}_{arm}_final"])
                torch.testing.assert_close(final,torch.stack((prior,m)).clamp(lower,upper),rtol=1e-5,atol=1e-6)
                scores = arrays[f"{name}_{arm}_final_scores"]
                winner = int(scores[1]<scores[0])
                assert row["arms"][arm]["winner"] == winner
                chosen = torch.from_numpy(arrays[f"{name}_{arm}_chosen"])
                torch.testing.assert_close(chosen,final[winner])
                raw = torch.from_numpy(arrays[f"{name}_{arm}_raw"])
                torch.testing.assert_close(raw,(chosen.reshape(5,2)*std+mean).clamp(-1,1),rtol=1e-5,atol=1e-6)


def test_saved_execution_and_evaluation_bindings():
    if not p.EVALUATION.exists():
        pytest.skip("physical evaluation not produced")
    context, decisions, executed, evaluated = (p.load_sealed(x) for x in
                                                (p.CONTEXT,p.DECISIONS,p.EXECUTION,p.EVALUATION))
    assert executed["context_sha256"] == p.sha(p.CONTEXT)
    assert executed["decisions_sha256"] == p.sha(p.DECISIONS)
    assert evaluated["execution_sha256"] == p.sha(p.EXECUTION)
    with np.load(p.CONTEXT.with_suffix(".npz"),allow_pickle=False) as ctx, \
         np.load(p.DECISIONS.with_suffix(".npz"),allow_pickle=False) as plan, \
         np.load(p.EXECUTION.with_suffix(".npz"),allow_pickle=False) as actual, \
         np.load(p.EVALUATION.with_suffix(".npz"),allow_pickle=False) as outcome:
        for i, row in enumerate(executed["rows"]):
            name = row["name"]
            for j,arm in enumerate(("one","three")):
                assert row["arms"][arm]["prefix_match_frames"] == 13
                actions = actual[f"{name}_{arm}_actions"]
                np.testing.assert_array_equal(actions[:12],ctx[name+"_actions"])
                np.testing.assert_array_equal(actions[12:],plan[f"{name}_{arm}_raw"])
                assert actual[f"{name}_{arm}_tail_frames"].shape == (5,224,224,3)
                distance = np.linalg.norm(outcome["endpoints"][2*i+j,:2]-np.array([1.8,2.8]))
                np.testing.assert_allclose(distance,evaluated["rows"][i][arm+"_distance"],rtol=0,atol=1e-12)
                assert evaluated["rows"][i][arm+"_success"] == bool(distance < .5)
    for stage, expected in (("context",{"reset_attempt":32,"reset_returned":32,"command_attempt":384,"command_returned":384}),
                            ("plan",{"encode_attempt":97,"encode_returned":97,"predict_attempt":256,"predict_returned":256}),
                            ("execute",{"reset_attempt":64,"reset_returned":64,"command_attempt":1088,"command_returned":1088}),
                            ("evaluate",{"endpoint_batch_attempt":1,"endpoint_batch_returned":1})):
        path = p.OUT.with_name(p.OUT.name+f"_{stage}_events.jsonl")
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert [e["index"] for e in events] == list(range(len(events)))
        counts = Counter(e["event"] for e in events)
        assert counts["completed"] == 1
        for event, total in expected.items():
            assert counts[event] == total
