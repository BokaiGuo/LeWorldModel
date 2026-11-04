import sys
import json

import numpy as np
import pytest
import torch

from experiments import pc_shadow_random_matched_dev_20261103_codex as module
from experiments import pc_shadow_spectral_closed_loop_dev_20261103_codex as base


def test_random_and_spectral_same_scoring_and_structure_budget():
    mean = torch.tensor([.2, -.1])
    std = torch.tensor([.5, 1.2])
    noise = torch.randn((3, 16, 10), generator=torch.Generator().manual_seed(31))
    response_count, logs = [], []
    def response(x):
        response_count.append(1)
        return torch.stack((x[0] + .5*x[2], x[1] - x[3]))
    def evaluate(x):
        return x.square().sum(dim=1)
    spectral, spec_structure, sc = base.decide("spectral", evaluate, response, mean, std, noise)
    with module.random_direction(noise, mean, std, lambda k, **v: logs.append((k, v))):
        random, rand_structure, rc = base.decide("spectral", evaluate, response, mean, std, noise)
    assert base.spectral_candidates is module.ORIGINAL
    assert sc == rc == dict(response_forward_calls=1, reverse_vector_calls=2, svd_calls=1, fd_model_forwards=0)
    assert len(response_count) == 2
    assert spectral["cem_counts"] == random["cem_counts"] == {"samples": 50, "batches": 4}
    assert torch.equal(random["trace"][0][0][1:5], rand_structure["normalized"])
    assert not torch.allclose(spec_structure["raw"], rand_structure["raw"])
    assert torch.allclose(rand_structure["raw"], torch.tensor([.25, -.25, .5, -.5])[:, None] * rand_structure["direction_raw"])
    assert torch.all(rand_structure["raw"].abs() <= 1)
    assert [k for k, _ in logs] == ["random_direction_returned"]


def test_random_direction_reproducible_and_independent_of_response():
    noise = torch.randn((3, 16, 10), generator=torch.Generator().manual_seed(2))
    mean, std = torch.zeros(2), torch.ones(2)
    values = []
    for response in (lambda x: torch.stack((x[0], x[1])),
                     lambda x: torch.stack((x[7], x[9]))):
        with module.random_direction(noise, mean, std, lambda *a, **k: None):
            _, structure, _ = base.decide("spectral", lambda x: x.square().sum(1),
                                          response, mean, std, noise)
            values.append(structure["raw"])
    assert torch.equal(*values)
    assert module.direction_seed(noise) != module.direction_seed(noise + .01)


def test_exception_restores_and_no_explicit_physics(monkeypatch, tmp_path):
    noise = torch.zeros(3, 16, 10)
    with pytest.raises(RuntimeError):
        with module.random_direction(noise, torch.zeros(2), torch.ones(2), lambda *a, **k: None):
            raise RuntimeError("test")
    assert base.spectral_candidates is module.ORIGINAL
    monkeypatch.setattr(module, "physical_run", lambda *a: pytest.fail("physics invoked"))
    monkeypatch.setattr(sys, "argv", ["test", "--output-dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path / "x").exists()


def test_paired_wiring_uses_independent_fake_rgb_and_separate_outputs(monkeypatch, tmp_path):
    seen = []
    class Bridge:
        def __init__(self, value):
            self.value = value
            self.step_count = 0
            self.initial_rgb = np.full((224, 224, 3), value, np.uint8)
        def step(self, action):
            self.step_count += 1
            return np.full((224, 224, 3), self.value+self.step_count, np.uint8)
        def close(self):
            pass
    def fake_physical(arms, directory):
        assert arms == ("spectral",)
        directory.mkdir()
        value = 10 if directory.name == "spectral" else 50
        def planner(previous, current, iteration):
            seen.append((directory.name, int(previous[0, 0, 0]), int(current[0, 0, 0])))
            noise = torch.zeros((3, 16, 10), dtype=torch.float32)
            result, _, _ = base.decide("spectral", lambda x: x.square().sum(1),
                lambda x: torch.stack((x[0], x[1])), torch.zeros(2), torch.ones(2), noise)
            return result["chosen"][:2].numpy().astype(np.float64), {}
        with base.DurableJournal(directory / "events.jsonl") as journal:
            base.run_arm("spectral", lambda: Bridge(value), np.zeros((1, 2)),
                         planner, directory, journal, iterations=2)
    monkeypatch.setattr(base, "physical_run", fake_physical)
    module.physical_run(tmp_path / "out")
    assert seen == [("spectral", 10, 11), ("spectral", 11, 12),
                    ("random", 50, 51), ("random", 51, 52)]
    events = [json.loads(line) for line in (tmp_path / "out/comparison_events.jsonl").read_text().splitlines()]
    assert [(e["event"], e.get("actual_arm")) for e in events[1:-1]] == [
        ("arm_attempt", "spectral"), ("arm_returned", "spectral"),
        ("arm_attempt", "random"), ("arm_returned", "random")]
    assert base.spectral_candidates is module.ORIGINAL


def test_random_failure_preserves_partial_result_and_restores(monkeypatch, tmp_path):
    original = base.decide
    def fake(arms, directory):
        directory.mkdir()
        if directory.name == "random":
            raise RuntimeError("uncertain")
    monkeypatch.setattr(base, "physical_run", fake)
    with pytest.raises(RuntimeError, match="uncertain"):
        module.physical_run(tmp_path / "out")
    assert base.decide is original
    assert (tmp_path / "out/spectral").is_dir()
    assert (tmp_path / "out/random").is_dir()
    events = [json.loads(line) for line in (tmp_path / "out/comparison_events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "comparison_failed"
