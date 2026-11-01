from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_root1_robust_feedback_20261101_exclusive as spec
from experiments.spectral_reachability import codex_p1_root1_robust_feedback_verify_20261101_exclusive as verify


def test_root1_rebind_and_all_prefixes_are_distinct_slots_not_new_roots():
    prepared, data, root, task = spec.inputs()
    assert root["goal_rgb_sha256"] == spec.CANDIDATES["goal_binding"]["goal_pixel_sha256"]
    assert task["binding"]["root_sha256"] == spec.CANDIDATES["goal_binding"]["root_sha256"]
    assert data["references"][1]["goal"].tolist() == root["goal"]
    assert spec.ROLES == ("selected", "zero", "greedy")
    assert prepared["source"]["expected_model_hash"] == spec.CANDIDATES["model"]["original_state_dict"]


def test_root1_stop_and_pair_accounting():
    assert verify.expected_neural(5) == {"actual_rgb_encode": 0, "action_encoder": 0, "predict": 0}
    assert verify.expected_neural(10) == {"actual_rgb_encode": 1, "action_encoder": 4, "predict": 4}
    assert verify.expected_neural(15) == {"actual_rgb_encode": 2, "action_encoder": 14, "predict": 14}
    verify.check_frame_count(7, 7, "censored_out_of_domain", 7)
    point = {k: "same" for k in spec.IDENTITY}
    row = {"observations": [point] * 11, "executed_raw": [[0, 0]] * 10}
    verify.check_matched_cp10(row, deepcopy(row))
    altered = deepcopy(row)
    altered["observations"][10]["snapshot"] = "different"
    with pytest.raises(ValueError):
        verify.check_matched_cp10(row, altered)


def test_contract_binds_runtime_helpers():
    from pathlib import Path
    from experiments.spectral_reachability import codex_p1_root1_robust_feedback_runner_20261101_exclusive as runner
    bindings = spec.contract()["bindings"]
    for module in (runner.old, runner.old.fixed, runner.engine, runner.risk, runner.physical.locking):
        assert str(Path(module.__file__).resolve()) in bindings


def test_same_episode_seal_and_stop_guards():
    from tests.test_codex_p1_cp10_rgb_refresh_verify_20261101_exclusive import events_for
    events = events_for("selected_nominal", 15)
    verify.check_order(events, "selected_nominal", 15)
    with pytest.raises(ValueError):
        verify.check_order([e for e in events if not (e["kind"] == "decision_sealed" and e["boundary"] == 10)],
                           "selected_nominal", 15)
    verify.check_order(events_for("selected_nominal", 7), "selected_nominal", 7)
    with pytest.raises(ValueError):
        verify.check_order(events_for("selected_nominal", 7) +
                           [{"kind": "branch_dispatch", "role": "selected_nominal", "step": 8, "command": [0, 0]}],
                           "selected_nominal", 7)


def test_root1_feedback_uses_actual_warmup_prefix_and_rgb():
    import numpy as np
    from experiments.spectral_reachability import codex_p1_root1_robust_feedback_runner_20261101_exclusive as r
    _, data, _, task = spec.inputs()
    prefix = np.asarray(spec.CANDIDATES["candidates"]["zero"]["prefix"], np.float32)
    warm = np.asarray(data["plan"]["raw_warmup"], np.float32)[5:]
    z = np.full(192, 0.25, np.float32)
    local = r.old.fixed.feedback_task(task, prefix, prefix, warm, z, {"boundary": 5})
    mean, std = np.asarray(task["action_mean"], np.float32), np.asarray(task["action_std"], np.float32)
    assert np.array_equal(np.asarray(local["root_context"]["action_history"], np.float32),
                          ((np.concatenate((warm, prefix)) - mean) / std).reshape(2, 10))
    assert local["root_context"]["latent_history"][-1] == z.tolist()
    assert local["goal"] == task["goal"]
    with pytest.raises(ValueError):
        r.old.fixed.feedback_task(task, prefix, prefix + np.float32(0.1), warm, z, {"boundary": 5})
