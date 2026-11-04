import numpy as np

from experiments import a0_pointmaze_packet_oracle_20261102 as source


def test_oracle_repeats_five_actions_and_compares_same_budget():
    bank = np.stack((np.linspace(-1, 1, 32), np.zeros(32)), axis=1)
    inputs = [{"root": 1, "bank": bank, "bank_sha256": "fixture",
               "source_state": np.zeros(4), "target": np.array([1.0, 0.0])}]
    seen = []

    def replay(states, actions):
        seen.append(actions.copy())
        assert actions.shape[1] in (5, 10)
        assert np.all(actions[:, :5] == actions[:, :1])
        if actions.shape[1] == 10:
            assert np.all(actions[:, 5:] == actions[:, 5:6])
        result = states.copy()
        result[:, :2] += actions.sum(axis=1)
        return result

    result = source.evaluate(inputs, replay)
    assert [len(x) for x in seen] == [32, 1024]
    assert result[0]["terminal_cost_at_terminal_choice"] <= result[0]["terminal_cost_with_prefix_first_best_tail"]
