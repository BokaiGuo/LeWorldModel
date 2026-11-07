import hashlib

import numpy as np
import pytest
import torch

from experiments.pc_pusht_all_shared_prefix_evaluate_20261107_codex import evaluate_plan


def fixture():
    actions = np.zeros((1, 2, 5, 2), dtype=np.float32)
    actions[0, 1, 2:] = 1
    digests = [hashlib.sha256(action.tobytes()).hexdigest() for action in actions[0]]
    plan = {"task": 0, "candidates": [{"call": 0, "slot": i, "sha256": digest}
                                      for i, digest in enumerate(digests)],
            "original_selected_position": 0, "selected_sha256": digests[0],
            "rescore_winner": 1, "winner_sha256": digests[1],
            "model_packet_evaluations": 6, "predictor_seconds": 0.1}
    physical = {"selected_index": 0, "records": [
        {"index": 0, "action_sha256": digests[0], "first_success_step": None, "final_success": False},
        {"index": 1, "action_sha256": digests[1], "first_success_step": 20, "final_success": True}]}
    return plan, np.array([2.0, 1.0]), [{"actions": torch.tensor(actions)}], physical


def test_selection_and_outcomes():
    result = evaluate_plan(*fixture())
    assert result["rescore"]["final_success"]
    assert not result["original"]["final_success"]
    assert result["best_ever_success_rank"] == 1


def test_rejects_wrong_choice():
    plan, scores, batches, physical = fixture()
    with pytest.raises(ValueError, match="argmin"):
        evaluate_plan(plan, scores[::-1], batches, physical)


def test_rejects_nonshared_prefix_even_with_matching_hash():
    plan, scores, batches, physical = fixture()
    batches[0]["actions"][0, 1, 0, 0] = 3
    digest = hashlib.sha256(batches[0]["actions"][0, 1].numpy().tobytes()).hexdigest()
    plan["candidates"][1]["sha256"] = plan["winner_sha256"] = digest
    physical["records"][1]["action_sha256"] = digest
    with pytest.raises(ValueError, match="prefix"):
        evaluate_plan(plan, scores, batches, physical)
