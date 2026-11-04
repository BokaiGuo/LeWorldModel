"""Independent, read-only checks for the sealed affine-inverse analysis."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.pc_pointmaze_safe_ranking_20261103 import ranking as safe_ranking


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "experiments/pc_pointmaze_affine_inverse_20261103.py"
spec = importlib.util.spec_from_file_location("affine_inverse_under_test", SCRIPT)
inverse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inverse)


@pytest.fixture
def bank():
    # Orthogonal columns make the expected inverse and residual explicit.
    weight = np.zeros((16, 4))
    weight[:4, :] = np.diag([2.0, 3.0, 4.0, 5.0])
    goal = np.tile(np.arange(16, dtype=float), (256, 1))
    delta = np.column_stack(
        (np.arange(48) / 10, -np.arange(48) / 20, np.ones(48), -np.ones(48))
    )
    predicted = goal[None, :, :] + (delta @ weight.T)[:, None, :]
    return predicted, goal, weight, np.array([2.0, 3.0, 4.0, 5.0]), delta


def test_affine_inverse_recovers_physical_position_distance(bank):
    predicted, goal, weight, std, delta = bank
    scores, residual, spread = inverse.project_tokens(predicted, goal, weight, std)
    expected = np.hypot(delta[:, 0] * std[0], delta[:, 1] * std[1])
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    np.testing.assert_allclose(spread, 0, atol=1e-12)


def test_perturbations_separate_column_space_residual_and_patch_spread(bank):
    predicted, goal, weight, std, _ = bank
    baseline = inverse.project_tokens(predicted, goal, weight, std)
    perturbed = predicted.copy()
    # This direction is orthogonal to the weight columns.
    perturbed[0, :, 8] += 3
    # Equal and opposite patch perturbations preserve the mean.
    perturbed[1, :128, 9] += 2
    perturbed[1, 128:, 9] -= 2
    scores, residual, spread = inverse.project_tokens(perturbed, goal, weight, std)
    np.testing.assert_allclose(scores, baseline[0], atol=1e-12)
    assert residual[0] == pytest.approx(3)
    assert residual[1] == pytest.approx(0, abs=1e-12)
    assert spread[0] == pytest.approx(0, abs=1e-12)
    assert spread[1] == pytest.approx(0.5)


def test_common_affine_bias_and_patch_permutation_do_not_change_results(bank):
    predicted, goal, weight, std, _ = bank
    baseline = inverse.project_tokens(predicted, goal, weight, std)
    bias = np.arange(16) * 1.25
    changed = inverse.project_tokens(
        (predicted + bias)[:, ::-1, :], (goal + bias)[::-1], weight, std
    )
    for actual, expected in zip(changed, baseline):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("index", range(4))
def test_rejects_each_invalid_shape(bank, index):
    args = list(bank[:4])
    args[index] = args[index][:-1]
    with pytest.raises(ValueError, match="contract"):
        inverse.project_tokens(*args)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_values_in_every_input(bank, index, value):
    args = [x.copy() for x in bank[:4]]
    args[index].flat[0] = value
    with pytest.raises(ValueError, match="contract"):
        inverse.project_tokens(*args)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_rejects_nonpositive_standard_deviation(bank, value):
    args = list(bank[:4])
    args[3] = args[3].copy()
    args[3][2] = value
    with pytest.raises(ValueError, match="contract"):
        inverse.project_tokens(*args)


def test_ranking_counts_overlap_ties_and_reversals():
    first = np.arange(48, dtype=float)
    second = -first.copy()
    first[1] = first[0]
    second[1] = second[0]
    second[3] = second[2]
    assert inverse.ranking(first, second) == {
        "comparable": 1126, "reversed": 1126, "first_ties": 1, "second_ties": 2
    }
    assert inverse.ranking(np.zeros(48), np.zeros(48)) == {
        "comparable": 0, "reversed": 0, "first_ties": 1128, "second_ties": 1128
    }


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_ranking_rejects_nonfinite_and_bad_shape(index, invalid):
    args = [np.arange(48, dtype=float), np.arange(48, dtype=float)]
    args[index][0] = invalid
    with pytest.raises(ValueError):
        inverse.ranking(*args)
    args[index] = np.zeros(47)
    with pytest.raises(ValueError):
        inverse.ranking(*args)


def test_ranking_reversals_do_not_underflow():
    first = np.arange(48, dtype=float) * 1e-200
    # Keep the sealed historical implementation intact and document its bug.
    assert inverse.ranking(first, -first)["reversed"] == 0
    result = safe_ranking(first, -first)
    assert result["comparable"] == 1128
    assert result["reversed"] == 1128


def test_safe_ranking_avoids_overflow_and_counts_ties():
    first = np.array([-1e308, 0.0, 1e308, 1e308])
    assert safe_ranking(first, -first) == {
        "comparable": 5, "reversed": 5, "first_ties": 1, "second_ties": 1
    }


def test_saved_scores_and_evaluation_read_only():
    score_path = inverse.OUT.with_name(inverse.OUT.name + "_scores.npz")
    receipt_path = inverse.OUT.with_name(inverse.OUT.name + "_scores.json")
    evaluation_path = inverse.OUT.with_name(inverse.OUT.name + "_evaluation.json")
    needed = [score_path, receipt_path, evaluation_path, inverse.RUN / "report.json",
              inverse.RUN / "evaluation.npz", inverse.CHECKPOINT, inverse.PROTOCOL]
    needed += [inverse.RUN / (key + ".npz") for key in inverse.GROUPS]
    if not all(path.is_file() for path in needed):
        pytest.skip("sealed source or output artifacts unavailable")

    receipt = json.loads(receipt_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    report = inverse.source_report()
    assert inverse.sha(score_path) == receipt["score_sha256"]
    assert inverse.sha(SCRIPT) == receipt["script_sha256"]
    assert inverse.sha(inverse.PROTOCOL) == receipt["protocol_sha256"]
    assert inverse.sha(inverse.RUN / "evaluation.npz") == report["evaluation_sha256"]
    assert inverse.sha(receipt_path) == evaluation["score_receipt_sha256"]
    weight, bias, _ = inverse.get_encoder()
    rows = {row["group"]: row for row in evaluation["rows"]}
    assert set(rows) == set(inverse.GROUPS)
    with np.load(score_path, allow_pickle=False) as saved, \
            np.load(inverse.RUN / "evaluation.npz", allow_pickle=False) as labels:
        for key in inverse.GROUPS:
            path = inverse.RUN / (key + ".npz")
            assert inverse.sha(path) == report["group_sha256"][key]
            with np.load(path, allow_pickle=False) as group:
                goal = group["goal_proprio_token"].reshape(256, 16).astype(float)
                state = group["goal_state_rgb_public"]
                expected_goal = ((state - inverse.MEAN) / inverse.STD) @ weight.T + bias
                np.testing.assert_allclose(goal, np.broadcast_to(expected_goal, goal.shape),
                                           rtol=0, atol=3e-5)
                for mode in inverse.MODES:
                    predicted = group[mode + "_predicted_proprio"].astype(float)
                    mean_delta = predicted.mean(axis=1) - goal.mean(axis=0)
                    # Solve independently, not through project_tokens or its pseudoinverse.
                    delta = np.linalg.lstsq(weight, mean_delta.T, rcond=None)[0].T
                    score = np.linalg.norm(delta[:, :2] * inverse.STD[:2], axis=1)
                    residual = np.linalg.norm(mean_delta - delta @ weight.T, axis=1)
                    spread = np.sqrt(np.mean(np.var(predicted, axis=1), axis=1))
                    for name, expected in (("score", score), ("residual", residual),
                                           ("patch_spread", spread)):
                        np.testing.assert_allclose(saved[f"{key}_{mode}_{name}"], expected,
                                                   rtol=1e-9, atol=1e-10)
                    row = rows[key]["modes"][mode]["affine_inverse"]
                    physical = labels[key + "_physical_distance"]
                    choice = int(np.argmin(score))
                    assert row["choice"] == choice
                    assert row["choice_physical_distance"] == pytest.approx(physical[choice])
                    for field, reference in (("vs_physical", physical),
                                             ("vs_actual_rgb", group["actual_rgb_cost"])):
                        pairs = [(np.sign(score[i] - score[j]),
                                  np.sign(reference[i] - reference[j]))
                                 for i in range(48) for j in range(i + 1, 48)]
                        expected = {
                            "comparable": sum(a != 0 and b != 0 for a, b in pairs),
                            "reversed": sum(a * b < 0 for a, b in pairs),
                            "first_ties": sum(a == 0 for a, _ in pairs),
                            "second_ties": sum(b == 0 for _, b in pairs),
                        }
                        assert row[field] == expected
                    for variant, values in (("affine_inverse", score),
                                            ("visual", group[mode + "_visual"]),
                                            ("proprio", group[mode + "_proprio"])):
                        stored = rows[key]["modes"][mode][variant]
                        assert safe_ranking(values, physical) == stored["vs_physical"]
                        assert safe_ranking(values, group["actual_rgb_cost"]) == stored["vs_actual_rgb"]
