import json

import numpy as np
import pytest
import torch

from experiments import pc_pointmaze_cp12_history_task_rank_20261103_codex as audit


def test_ranking_reversals_and_ties():
    x = np.arange(8.)
    assert audit.ranking(x, -x) == {"pairs": 28, "comparable": 28, "reversed": 28,
                                    "left_ties": 0, "right_ties": 0}
    x[1] = x[0]
    result = audit.ranking(x, np.arange(8.))
    assert result == {"pairs": 28, "comparable": 27, "reversed": 0,
                      "left_ties": 1, "right_ties": 0}
    with pytest.raises(ValueError):
        audit.ranking([0] * 7, [0] * 7)
    x[0] = np.nan
    with pytest.raises(ValueError):
        audit.ranking(x, np.arange(8.))


def test_projected_cost_known_affine_geometry():
    # An identity-position embedding must return Euclidean position distance,
    # irrespective of offsets in the two velocity coordinates.
    pinv = torch.zeros((4, 16))
    pinv[:4, :4] = torch.eye(4)
    goal = torch.zeros((256, 16))
    tokens = torch.zeros((1, 256, 16))
    tokens[:, :, :4] = torch.tensor([3., 4., 100., -100.])
    cost = audit.h.projected_cost(tokens, goal, pinv, torch.ones(2))
    assert cost.item() == 5.


def test_no_overwrite_before_reading_inputs(monkeypatch, tmp_path):
    occupied = tmp_path / "occupied"
    occupied.touch()
    monkeypatch.setattr(audit, "SCORE", occupied)
    with pytest.raises(FileExistsError):
        audit.score_stage()
    monkeypatch.setattr(audit, "EVAL", occupied)
    with pytest.raises(FileExistsError):
        audit.eval_stage()


def test_saved_scores_and_rankings_independently():
    if not audit.EVAL.exists():
        pytest.skip("saved diagnostic has not been run")
    score = json.loads(audit.SCORE.read_text())
    result = json.loads(audit.EVAL.read_text())
    old = json.loads(audit.REFRESH.read_text())
    assert audit.sha(audit.SCORE) == result["score_receipt_sha256"]
    assert audit.sha(audit.SCORE_NPZ) == score["score_arrays_sha256"]
    assert audit.sha(audit.PROTOCOL) == score["protocol_sha256"] == result["protocol_sha256"]
    assert score["counts"] == {"projected_scores": 96, "goal_affine_encodes": 1,
                               "model_prediction_samples": 0, "image_encodes": 0,
                               "physical_steps": 0, "candidate_selections": 0}
    weight, _, _ = audit.h.get_encoder()
    pinv = np.linalg.pinv(weight).astype(np.float32)
    with np.load(audit.SCORE_NPZ, allow_pickle=False) as out, \
            np.load(audit.PRED.with_suffix(".npz"), allow_pickle=False) as pred:
        goal = out["goal_token"].mean(axis=0)
        for row in score["rows"]:
            name = f"r{row['root']}_a{('raw', 'spectral').index(row['arm'])}_p{row['candidate']}"
            for arm in ("one", "three", "target"):
                tokens = pred[f"{name}_{arm}_proprio"].reshape(256, 16)
                delta = (tokens.mean(axis=0) - goal) @ pinv.T
                calculated = np.linalg.norm(delta[:2] * audit.h.STD[:2])
                np.testing.assert_allclose(calculated, row[arm], rtol=3e-5, atol=1e-6)
    for k, row in enumerate(result["rows"]):
        group = score["rows"][8*k:8*k+8]
        prior = old["rows"][k]
        assert row["root"] == prior["root"] == audit.ROOTS[k]
        for arm in ("one", "three"):
            for ref, field in (("rgb", "actual_score"), ("physical", "physical_distance")):
                # Independent simple pairwise loop, rather than the production rank helper.
                left = [r[arm] for r in group]
                right = [r[field] for r in prior["branches"]]
                pairs = [(left[i] - left[j], right[i] - right[j])
                         for i in range(8) for j in range(i+1, 8)]
                expected = {"pairs": 28, "comparable": sum(x != 0 and y != 0 for x, y in pairs),
                            "reversed": sum(x*y < 0 for x, y in pairs),
                            "left_ties": sum(x == 0 for x, _ in pairs),
                            "right_ties": sum(y == 0 for _, y in pairs)}
                assert row["rankings"][f"{arm}_vs_{ref}"] == expected
        for new, previous in zip(group, prior["branches"], strict=True):
            np.testing.assert_allclose(new["one"], previous["refresh_score"], rtol=3e-5, atol=1e-6)
            np.testing.assert_allclose(new["target"], previous["actual_score"], rtol=3e-5, atol=1e-6)
