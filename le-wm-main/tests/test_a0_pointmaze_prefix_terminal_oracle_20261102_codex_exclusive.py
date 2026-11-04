import copy
import json

import numpy as np
import pytest

from experiments import a0_pointmaze_prefix_terminal_oracle_20261102_codex_exclusive as audit


def test_source_identity_and_real_selected_endpoint_replay():
    assert audit.digest(audit.SOURCE) == audit.SOURCE_SHA
    items = audit.extract_inputs(json.loads(audit.SOURCE.read_bytes()))
    result = json.loads((audit.ROOT / "results" /
                         (audit.NAME + "_RESULT01.json")).read_bytes())
    assert result["summary"] == {"roots": 8, "different_first_action": 2,
                                "success_rate": None, "closed_loop_gain": None}
    for item, row in zip(items, result["rows"], strict=True):
        i, j = row["terminal_selected_indices"]
        k = row["prefix_selected_index"]
        sources = np.repeat(item["source_state"][None, :], 2, axis=0)
        actions = np.array([[item["bank"][i], item["bank"][j]],
                            [item["bank"][k], item["bank"][k]]])
        actual = audit.physical.replay_position_endpoints(sources, actions)
        assert np.allclose(actual[0, :2], row["terminal_positions"][i][j], atol=1e-12, rtol=0)
        assert np.allclose(actual[1, :2], row["terminal_positions"][k][k], atol=1e-12, rtol=0)
        prefix_costs = np.linalg.norm(np.asarray(row["prefix_positions"]) - item["target"], axis=1)
        terminal_costs = np.linalg.norm(np.asarray(row["terminal_positions"]) - item["target"], axis=2)
        assert int(np.argmin(prefix_costs)) == k
        assert np.unravel_index(np.argmin(terminal_costs), terminal_costs.shape) == (i, j)
    assert not result["protocol"]["rgb_only_planner"]
    assert not result["scientific_admission"]


def test_mutated_shared_bank_fails_closed():
    report = json.loads(audit.SOURCE.read_bytes())
    changed = copy.deepcopy(report)
    row = next(r for r in changed["rows"]
               if r["episode_id"] == 1900 and r["method"] == "perfect_dynamics_mpc")
    row["policy_contract"]["candidate_actions"][0][0] += 0.01
    with pytest.raises(ValueError, match="bank identity"):
        audit.extract_inputs(changed)


def test_replay_sequence_order_and_stable_tie():
    item = {"root": 1, "source_state": np.zeros(4), "target": np.zeros(2),
            "bank": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "bank_sha256": "fixture", "state_matches": {}}
    calls = []
    def replay(sources, sequences):
        calls.append(sequences.copy())
        result = np.zeros_like(sources)
        result[:, 0] = sequences[:, :, 0].sum(axis=1)
        return result
    row = audit.evaluate([item], replay)[0]
    assert calls[1].tolist() == [[[0, 0], [0, 0]], [[0, 0], [1, 0]],
                                  [[1, 0], [0, 0]], [[1, 0], [1, 0]]]
    assert row["prefix_selected_index"] == 0
    assert row["terminal_selected_indices"] == [0, 0]


def test_bad_physical_result_rejected():
    item = {"root": 1, "source_state": np.zeros(4), "target": np.zeros(2),
            "bank": np.zeros((2, 2)), "bank_sha256": "fixture", "state_matches": {}}
    with pytest.raises(ValueError, match="physical endpoint shape"):
        audit.evaluate([item], lambda sources, sequences: np.full(sources.shape, np.nan))
