import json

import numpy as np
import pytest

from experiments import pc_response_diverse_beam_exact_dev_20261103_codex as dev


def test_recorded_directions_retention_budget_and_selection():
    if not dev.PLANS.exists() or not dev.AUDIT.exists():
        pytest.skip("development artifacts unavailable")
    data = json.loads(dev.PLANS.read_text())
    audit = json.loads(dev.AUDIT.read_text())
    assert data["dependencies"] == dev.dependencies()
    assert audit["plans_sha256"] == dev.prior.sha(dev.PLANS)
    assert len(data["rows"]) == len(audit["rows"]) == 96
    for row in data["rows"]:
        result, task = row["result"], row["task"]
        assert sum(x["tokens"] for x in row["trace"]) == row["tokens"] <= 480
        assert result["selected"]["terminal_score"] == min(result["precomparison"]["terminal_score"],
                                                            result["comparison_score"])
        if row["arm"] == "full_cem":
            continue
        first = np.asarray([item["prefix"] for item in result["first_bank"]], dtype=np.float32)
        positions = dev.prior.exact.endpoints(task, first)
        scores = np.linalg.norm(positions - task["goal"], axis=1)
        np.testing.assert_allclose(positions, result["first_positions"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(scores, result["prefix_scores"], atol=1e-12, rtol=0)
        best = int(np.argmin(scores))
        if row["arm"] == "response_greedy":
            expected = np.argsort(scores, kind="stable")[:2].tolist()
        else:
            distances = np.linalg.norm(positions - positions[best], axis=1)
            distances[best] = -np.inf
            expected = [best, int(np.argmax(distances))]
        assert result["retained"] == expected
        assert len(result["bank"]) == len(result["programs"]) == 6
        assert result["precomparison"]["terminal_score"] <= min(x["terminal_score"] for x in result["programs"])
        for item in result["bank"]:
            np.testing.assert_array_equal(np.asarray(item["prefix"])[:5],
                                          result["first_bank"][item["parent_index"]]["prefix"])
        if row["arm"].startswith("response"):
            evidence = result["response"]
            nominal = np.asarray(evidence["nominal"])
            axes = np.eye(10).reshape(10, 5, 2)
            plus = (nominal + .05 * axes).astype(np.float32)
            minus = (nominal - .05 * axes).astype(np.float32)
            jacobian = (dev.prior.exact.endpoints(task, plus) - dev.prior.exact.endpoints(task, minus)).T / .1
            np.testing.assert_allclose(jacobian, evidence["jacobian"], atol=1e-12, rtol=0)
            _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
            np.testing.assert_allclose(singular, evidence["singular"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(vt[:2], evidence["right_directions"], atol=1e-12, rtol=0)
            assert result["structure_counts"]["svd"] == 3
        else:
            assert result["structure_counts"].get("svd", 0) == 0
