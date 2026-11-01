import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_host_cuda_20261030 as host
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as original
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from test_codex_four_arm_raw_candidates_20261030 import make


def args(leaf, cfg, policy, contract, arm):
    return (leaf, arm, torch.zeros(1, 1, 1), torch.empty(1, 0, 10), torch.ones(1, 1),
            torch.zeros(2), torch.ones(2), cfg, policy, contract)


@pytest.mark.parametrize("arm", ["structured", "native", "random", "immediate_progress"])
def test_all_candidates_saved_and_matches_existing_variant(tmp_path, arm):
    (tmp_path / "new").mkdir()
    (tmp_path / "old").mkdir()
    cfg, policy, contract, sha, d, leaf = make(tmp_path / "new", arm)
    cfg2, policy2, contract2, _, d2, leaf2 = make(tmp_path / "old", arm)
    events = []
    result = host.execute_arm(*args(leaf, cfg, policy, contract, arm), emit=lambda kind, data: events.append((kind, data)))
    prior = original.run_arm(*args(leaf2, cfg2, policy2, contract2, arm))
    assert torch.equal(result["selected_raw"], prior["selected_raw"])
    assert result["final_scores"] == prior["final_scores"]
    assert [kind for kind, _ in events].count("generation_candidates") == cfg["iterations"]
    assert [kind for kind, _ in events].count("generation_scores") == cfg["iterations"]
    assert all(len(data["raw"]) == 3 for kind, data in events if kind == "generation_candidates")
    assert [kind for kind, _ in events].count("final_selection") == 1
    if arm == "immediate_progress":
        assert any(kind == "screening_candidates" for kind, _ in events)
        assert sum(kind == "screening_score" for kind, _ in events) == 3
    if arm == "structured":
        assert any(kind == "structured_signal" for kind, _ in events)
    r = d.finish()
    consumed = consumer.consume(r["journal"], r["sha256"], sha, arm, "a", arm)
    assert consumed["status"] == "closed_unverified"
    d2.finish()


def test_candidate_persist_failure_stops_before_replay(tmp_path):
    cfg, policy, contract, sha, d, leaf = make(tmp_path, "native")
    def emit(kind, data):
        if kind == "generation_candidates":
            raise OSError("disk failure")
    with pytest.raises(OSError, match="disk failure"):
        host.execute_arm(*args(leaf, cfg, policy, contract, "native"), emit=emit)
    r = d.finish()
    observed = consumer.consume(r["journal"], r["sha256"], sha, "native", "a", "native")
    assert observed["status"] == "failed_or_blocked"
    assert observed["charged_upper_bounds"]["predict_calls"] == 0


def test_resource_math_and_wall_guard():
    caps = host.resource_caps(dim=192, horizon=3, population=3, iterations=2)
    assert caps["predict_calls"] == 30 and caps["backward_calls"] == 192
    assert caps["vjp_rows"] == 192 and caps["replay_calls"] == 12
    ticks = iter([0, 1, 5])
    guard = host.Guard(4, lambda: next(ticks), lambda: (0, 0), 10, 10)
    guard.check()
    with pytest.raises(ValueError, match="wall"):
        guard.check()


def test_no_cuda_without_explicit_authorization(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(host, "require_cuda_allocation", lambda *_: called.append(True))
    with pytest.raises(ValueError, match="authorization"):
        host.run_cuda(tmp_path / "bad", {}, "not-approved")
    assert not called


def test_cli_partial_is_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(host, "read_prepare", lambda *_: {})
    monkeypatch.setattr(host, "run_cuda", lambda *_: {"status": "failed_or_partial"})
    monkeypatch.setattr("sys.argv", ["entry", "run", "--prepared", "x", "--sha256", "x",
                                    "--output", "x", "--authorization", host.AUTH])
    assert host.main() == 1
    assert "failed_or_partial" in capsys.readouterr().out
