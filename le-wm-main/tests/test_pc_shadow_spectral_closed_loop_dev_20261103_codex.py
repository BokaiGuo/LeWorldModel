import json
import sys

import numpy as np
import pytest
import torch

from experiments import pc_shadow_spectral_closed_loop_dev_20261103_codex as dev


def test_policy_counts_candidates_and_degenerate():
    mean, std = torch.zeros(2), torch.ones(2)
    noise = torch.randn((3, 16, 10), generator=torch.Generator().manual_seed(42))
    seen = []

    def evaluate(x):
        seen.append(x.clone())
        return (x - .2).square().sum(1)

    def response(x):
        return torch.stack((x[0] + x[2], x[1]))

    baseline, structure, counts = dev.decide("baseline", evaluate, response, mean, std, noise)
    assert structure is None and counts["response_forward_calls"] == 0
    assert baseline["cem_counts"] == {"samples": 50, "batches": 4}
    seen.clear()
    result, structure, counts = dev.decide("spectral", evaluate, response, mean, std, noise)
    assert counts == {"response_forward_calls": 1, "reverse_vector_calls": 2, "svd_calls": 1,
                      "fd_model_forwards": 0}
    assert torch.equal(seen[0][1:5], structure["normalized"])
    assert result["cem_counts"] == {"samples": 50, "batches": 4}
    with pytest.raises(ValueError):
        dev.decide("spectral", evaluate, lambda x: torch.stack((x[0]*0, x[1]*0)),
                   mean, std, noise)


def test_independent_fake_streams_and_durable_events(tmp_path):
    class Fake:
        def __init__(self, value):
            self.step_count = 0
            self.initial_rgb = np.full((224, 224, 3), value, np.uint8)
            self.value = value

        def step(self, action):
            self.step_count += 1
            return np.full((224, 224, 3), self.value + self.step_count, np.uint8)

        def close(self):
            pass

    observations = []
    with dev.DurableJournal(tmp_path / "events.jsonl") as journal:
        for arm, value in [("baseline", 10), ("spectral", 50)]:
            def planner(previous, current, iteration):
                observations.append((arm, int(previous[0, 0, 0]), int(current[0, 0, 0])))
                return np.array([.1, 0.]) if arm == "baseline" else np.array([-.1, 0.]), {}
            dev.run_arm(arm, lambda: Fake(value), np.zeros((2, 2)), planner,
                        tmp_path, journal, iterations=2)
    assert observations == [("baseline", 11, 12), ("baseline", 12, 13),
                            ("spectral", 51, 52), ("spectral", 52, 53)]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["event"] == "bridge_attempt"]) == 10
    assert len([e for e in events if e["event"] == "bridge_returned"]) == 10
    assert all(e["snapshot_sha256"] for e in events if e["event"] == "bridge_attempt")


def test_failure_is_not_retried(tmp_path):
    class Failed:
        step_count = 0
        initial_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        def step(self, action):
            self.step_count += 1
            raise RuntimeError("unknown execution")
        def close(self):
            pass
    with dev.DurableJournal(tmp_path / "events.jsonl") as journal:
        with pytest.raises(RuntimeError, match="unknown execution"):
            dev.run_arm("baseline", Failed, np.zeros((1, 2)), lambda *a: None, tmp_path, journal, 1)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [x["event"] for x in events] == ["bridge_attempt", "bridge_returned", "bridge_attempt"]


def test_cli_requires_explicit_authorization(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(dev, "physical_run", lambda *args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["executor", "--output-dir", str(tmp_path / "never")])
    with pytest.raises(SystemExit):
        dev.main()
    assert called == []
    assert not (tmp_path / "never").exists()


def test_dispatch_rejects_snapshot_action_mismatch(tmp_path):
    snapshot = tmp_path / "snapshot.npz"
    digest = dev.save_snapshot(snapshot, clock=np.asarray(1), action=np.zeros(2))
    class Never:
        def advance(self, action):
            raise AssertionError("must not send")
    with dev.DurableJournal(tmp_path / "events.jsonl") as journal:
        with pytest.raises(ValueError, match="snapshot action"):
            dev.dispatch_one(Never(), np.ones(2), snapshot, digest, journal, tmp_path / "out",
                             "baseline", "feedback", 1)
    assert (tmp_path / "events.jsonl").read_text() == ""
