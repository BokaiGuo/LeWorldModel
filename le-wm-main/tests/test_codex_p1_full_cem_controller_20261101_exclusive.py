import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_p1_full_cem_controller_20261101_exclusive as m
from experiments.spectral_reachability.codex_e1_proposal_durable_events_20261101 import read_events


class Leaf:
    def action_encode(self, actions, *, phase):
        return actions

    def predict(self, states, actions, *, phase):
        result = states.clone()
        result[:, -1, :] += actions[:, -1, :].mean(dim=-1, keepdim=True)
        return result


def setup(tmp_path, monkeypatch, *, issue=None):
    monkeypatch.setattr(m, "ITERATIONS", 2)
    monkeypatch.setattr(m, "SAMPLES", 4)
    monkeypatch.setattr(m, "ELITES", 2)
    goal, matrix = torch.ones(1, 2), torch.eye(2)
    mean, std = torch.zeros(2), torch.ones(2)
    identity = dict(model_sha256="a" * 64, root_sha256="b" * 64,
                    task_sha256="c" * 64, goal_sha256=m.tensor_sha(goal),
                    metric_sha256=m.tensor_sha(matrix),
                    normalizer_sha256=m.tensor_sha(torch.stack((mean, std))))
    observations, packets = [], []

    def observe(boundary, leaf):
        observations.append(boundary)
        return dict(identity=identity.copy(), rgb_sha256="d" * 64,
                    provenance="synthetic fixture", encoding_mode="inherited", states=torch.zeros(1, 3, 2),
                    history=torch.zeros(1, 2, 10), goal=goal, matrix=matrix, mean=mean, std=std)

    def send(packet):
        packets.append(packet.copy())
        return issue(packet) if issue else {"executed_steps": 5, "stopped": False}

    path = tmp_path / "events.jsonl"
    controller = m.Controller(leaf=Leaf(), observe=observe, issue=send,
                              journal=path, run_id="test", identity=identity, seed=3)
    return controller, path, observations, packets


def entries(path):
    return [x["payload"]["event"] for x in read_events(path)]


def test_full_feedback_schedule_and_attempt_costs(tmp_path, monkeypatch):
    c, path, observations, packets = setup(tmp_path, monkeypatch)
    results = [c.step() for _ in range(4)]
    with pytest.raises(RuntimeError):
        c.step()
    c.close()
    assert observations == [0, 5, 10, 15]
    assert [r["horizon"] for r in results] == [3, 2, 1, 1]
    assert len(packets) == 4 and all(p.shape == (5, 2) and np.max(np.abs(p)) <= 1 for p in packets)
    events = entries(path)
    attempts = [e["operation"] for e in events if e["type"] == "attempt"]
    assert attempts.count("rollout") == 4 * 9
    assert attempts.count("predict") == attempts.count("action_encode") == 9 * 7
    assert attempts.count("issue_packet") == 4
    assert sum(e["type"] == "bank" for e in events) == 8
    assert sum(e["type"] == "winner_replay" for e in events) == 4


def test_partial_native_stop_is_retained_without_next_observation(tmp_path, monkeypatch):
    c, path, observations, _ = setup(tmp_path, monkeypatch,
                                     issue=lambda p: {"executed_steps": 2, "stopped": True})
    assert c.step()["receipt"]["executed_steps"] == 2
    with pytest.raises(RuntimeError):
        c.step()
    c.close()
    assert observations == [0]
    assert any(e.get("receipt", {}).get("executed_steps") == 2 for e in entries(path))


def test_predict_failure_is_durable_and_never_dispatches_action(tmp_path, monkeypatch):
    c, path, _, packets = setup(tmp_path, monkeypatch)
    c.leaf.predict = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model failed"))
    with pytest.raises(RuntimeError, match="model failed"):
        c.step()
    with pytest.raises(RuntimeError, match="stopped"):
        c.step()
    c.close()
    events = entries(path)
    assert not packets
    assert any(e["type"] == "failure" for e in events)
    assert events[-1]["type"] == "controller_failure"


def test_goal_mismatch_fails_before_model_or_actions(tmp_path, monkeypatch):
    c, path, _, packets = setup(tmp_path, monkeypatch)
    original = c.observe
    def bad(boundary, leaf):
        value = original(boundary, leaf)
        value["goal"] = torch.zeros(1, 2)
        return value
    c.observe = bad
    with pytest.raises(ValueError, match="goal or metric"):
        c.step()
    c.close()
    assert not packets
    assert not any(e.get("operation") == "predict" for e in entries(path))


def test_existing_journal_not_overwritten(tmp_path, monkeypatch):
    c, path, _, _ = setup(tmp_path, monkeypatch)
    c.close()
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        m.Controller(leaf=Leaf(), observe=None, issue=None, journal=path,
                     run_id="other", identity=c.identity, seed=3)
    assert path.read_bytes() == old


def test_fresh_encoding_is_metered_before_observation_returns(tmp_path, monkeypatch):
    c, path, _, _ = setup(tmp_path, monkeypatch)
    original = c.observe
    c.leaf.encode = lambda info, *, phase: torch.tensor([1.])

    def observe(boundary, leaf):
        leaf.encode({}, phase="setup")
        value = original(boundary, leaf)
        value["encoding_mode"] = "fresh_metered"
        return value

    c.observe = observe
    c.step()
    c.close()
    attempts = [e.get("operation") for e in entries(path) if e["type"] == "attempt"]
    assert attempts[:2] == ["observe", "encode"]
    assert attempts.count("encode") == 1


def test_encoding_claim_without_metered_call_fails(tmp_path, monkeypatch):
    c, path, _, packets = setup(tmp_path, monkeypatch)
    original = c.observe

    def observe(boundary, leaf):
        value = original(boundary, leaf)
        value["encoding_mode"] = "fresh_metered"
        return value

    c.observe = observe
    with pytest.raises(ValueError, match="encoding mode"):
        c.step()
    c.close()
    assert not packets


def test_failed_issue_is_recorded_and_never_retried(tmp_path, monkeypatch):
    def issue(packet):
        raise RuntimeError("unknown action delivery")
    c, path, _, packets = setup(tmp_path, monkeypatch, issue=issue)
    with pytest.raises(RuntimeError, match="unknown action delivery"):
        c.step()
    with pytest.raises(RuntimeError, match="stopped"):
        c.step()
    c.close()
    events = entries(path)
    assert len(packets) == 1
    assert sum(e.get("operation") == "issue_packet" for e in events) == 1
    assert any(e["type"] == "failure" and e["message"] == "unknown action delivery" for e in events)
