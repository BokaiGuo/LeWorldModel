import pytest
from copy import deepcopy

from experiments.spectral_reachability import codex_p1_cp10_robust_feedback_20261101_exclusive as m


def test_cp5_policies_use_same_fixed_q1_rule():
    q1, q2 = [0.3, 0.5], [0.6, 0.2]
    assert m.select_cp5("nominal", q1, q2) == "hold"
    assert m.select_cp5("robust", q1, q2) == "hold"
    assert m.select_cp5("nominal", [0.1, 0.1], q2) == "hold"
    with pytest.raises(ValueError):
        m.select_cp5("unknown", q1, q2)
    with pytest.raises(ValueError):
        m.select_cp5("nominal", [float("nan"), 1], q2)


def test_cp10_nominal_and_robust_differ_and_tie_hold():
    scores = [[0.1] * 5, [0.05, 0.05, 0.05, 0.05, 0.5]]
    assert m.select_cp10("nominal", scores) == "optimized"
    assert m.select_cp10("robust", scores) == "hold"
    assert m.select_cp10("robust", [[0.1] * 5, [0.1] * 5]) == "hold"
    with pytest.raises(ValueError):
        m.select_cp10("robust", [[0.1], [0.2]])


def test_pair_check_rejects_different_first10_or_snapshot():
    from experiments.spectral_reachability import codex_p1_cp10_robust_feedback_verify_20261101_exclusive as v
    point = {key: "same" for key in m.IDENTITY}
    row = {"observations": [point] * 11, "executed_raw": [[0, 0]] * 10}
    v.check_matched_cp10(row, deepcopy(row))
    changed = deepcopy(row)
    changed["observations"][10]["snapshot"] = "different"
    with pytest.raises(ValueError):
        v.check_matched_cp10(row, changed)
    changed = deepcopy(row)
    changed["executed_raw"][5] = [1, 0]
    with pytest.raises(ValueError):
        v.check_matched_cp10(row, changed)


def test_protocol_is_nonconfirmatory_and_equal_budget():
    value = m.protocol()
    assert value["neural_caps_per_complete_episode"] == {"actual_rgb_encode": 2, "action_encoder": 14, "predict": 14}
    assert value["physical_future_used_for_selection"] is False
    assert value["outcome_observed_before_design"] is True
    assert value["P1_admission"] is False


def test_one_rollout_yields_both_horizons_without_extra_forward():
    import numpy as np
    import torch
    from experiments.spectral_reachability import codex_p1_cp10_robust_feedback_runner_20261101_exclusive as runner

    class Fake:
        def action_encoder(self, context):
            return context

        def predict(self, states, actions):
            return states + 1

    class Meter:
        def __init__(self):
            self.calls = []

        def call(self, kind, context, fn):
            self.calls.append(kind)
            return fn()

    task = {"root_context": {"latent_history": np.zeros((3, 192)).tolist(),
                             "action_history": np.zeros((2, 10)).tolist()},
            "goal": np.zeros(192).tolist(), "action_mean": [0, 0], "action_std": [1, 1]}
    meter = Meter()
    values = runner.score(Fake(), torch.device("cpu"), task, np.zeros((2, 5, 2)), meter, 5, "hold")
    assert values == [1.0, 4.0]
    assert meter.calls == ["action_encoder", "predict", "action_encoder", "predict"]


def test_separate_preseal_is_exclusive_and_does_not_run(monkeypatch, tmp_path):
    from experiments.spectral_reachability import codex_p1_cp10_robust_feedback_runner_20261101_exclusive as runner
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "activation", lambda: {"fixed": True})
    path, digest = runner.prepare("TEST01")
    assert runner.sha(path) == digest
    with pytest.raises((ValueError, FileExistsError, AssertionError)):
        runner.prepare("TEST01")
    assert not runner.output_path("TEST01").exists()


def test_new_verifier_rejects_missing_seal_and_bad_accounting():
    from experiments.spectral_reachability import codex_p1_cp10_robust_feedback_verify_20261101_exclusive as v
    from tests.test_codex_p1_cp10_rgb_refresh_verify_20261101_exclusive import events_for
    events = events_for("selected_q1_feedback")
    v.check_order(events, "selected_q1_feedback", 15)
    with pytest.raises(ValueError):
        v.check_order([e for e in events if not (e["kind"] == "decision_sealed" and e["boundary"] == 10)],
                      "selected_q1_feedback", 15)
    pair = [{"kind": "neural_attempt", "role": "a", "resource": "predict", "context": {"boundary": 5}},
            {"kind": "neural_return", "role": "a", "resource": "predict", "context": {"boundary": 5}}]
    counts = {"actual_rgb_encode": 0, "action_encoder": 0, "predict": 1}
    v.check_neural(pair, "a", counts, counts)
    bad = deepcopy(pair)
    bad[1]["context"]["boundary"] = 10
    with pytest.raises(ValueError):
        v.check_neural(bad, "a", counts, counts)
    with pytest.raises(ValueError):
        v.check_neural(pair[:1], "a", counts, counts)
    assert v.expected_neural(10) == {"actual_rgb_encode": 1, "action_encoder": 4, "predict": 4}
    assert v.expected_neural(15) == {"actual_rgb_encode": 2, "action_encoder": 14, "predict": 14}
    assert v.unobserved(15, 10) == 5
    v.check_frame_count(7, 7, "censored_out_of_domain", 7)
    with pytest.raises(ValueError):
        v.check_frame_count(7, 7, "native_stopped", None)
