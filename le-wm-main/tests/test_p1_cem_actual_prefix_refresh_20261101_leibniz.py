from contextlib import contextmanager
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_cem_actual_prefix_refresh_20261101_leibniz as subject


class Analytic:
    def __init__(self):
        self.contexts = []
        self.actions = []

    def action_encode(self, action, *, phase):
        self.actions.append(action.clone())
        return action

    def predict(self, states, action, *, phase):
        self.contexts.append(states.clone())
        return states + action[..., :1]


def test_fixed_actions_and_actual_prefix_sliding_history():
    leaf = Analytic()
    states = torch.zeros((1, 3, 192))
    history = torch.zeros((1, 2, 10))
    raw = torch.tensor([[[.1, .1]] * 5, [[.2, .2]] * 5, [[.3, .3]] * 5])
    actual = torch.stack([torch.full((192,), float(i)) for i in range(4)])
    result = subject.simulate(leaf, states, history, raw, actual, torch.zeros(2), torch.ones(2))
    assert np.array_equal(result["open"][0], result["actual_prefix"][0])
    assert torch.equal(leaf.contexts[4][0, -1], actual[1])
    assert torch.equal(leaf.contexts[5][0, -2], actual[1])
    assert torch.equal(leaf.contexts[5][0, -1], actual[2])
    for i in range(3):
        assert torch.equal(leaf.actions[i], leaf.actions[i + 3])
    assert not np.array_equal(result["open"][2], result["actual_prefix"][2])


def test_reject_unbounded_and_nan():
    leaf = Analytic()
    with pytest.raises(ValueError):
        subject.simulate(leaf, torch.zeros(1, 3, 192), torch.zeros(1, 2, 10),
                         torch.full((3, 5, 2), 2.), torch.zeros(4, 192),
                         torch.zeros(2), torch.ones(2))
    assert not leaf.actions


def test_real_run_lock_path_pre_cuda(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    source = tmp_path / "source"
    source.write_text("source")
    value = {"sources": {str(source): subject.family.sha(source)}}
    prepared = results / (subject.NAME + "_PREP.json")
    prepared.write_text(json.dumps(value))
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "build", lambda: value)
    seen = []
    @contextmanager
    def lock(path):
        seen.append(path)
        yield
    monkeypatch.setattr(subject.prev.launcher.extra.gpu.locking, "gpu_lock", lock)
    def no_cuda(_):
        raise RuntimeError("before CUDA")
    monkeypatch.setattr(subject.own.gpu_preflight, "require_cuda_allocation", no_cuda)
    output = results / (subject.NAME + "_RUN")
    result = subject.run(prepared, subject.family.sha(prepared), output, subject.AUTH)
    assert seen == [subject.prev.launcher.extra.gpu.LOCK]
    assert result["status"] == "partial"
    assert json.loads((output / "report.json").read_text())["failure"]["message"] == "before CUDA"
