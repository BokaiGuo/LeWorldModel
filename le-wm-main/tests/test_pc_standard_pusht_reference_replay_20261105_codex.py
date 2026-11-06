import numpy as np

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def test_panel_matches_native_sampling_including_last_row_exclusion():
    episodes = np.repeat(np.arange(4), 30)
    steps = np.tile(np.arange(30), 4)
    data = {"episode_idx": episodes, "step_idx": steps}
    valid = np.flatnonzero(steps <= 30 - 25 - 1)
    expected = np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, size=5, replace=False)])
    assert np.array_equal(panel_rows(data, count=5), expected)
    assert valid[-1] not in expected


def test_trace_values_are_strict_json_compatible():
    import json
    from experiments.pc_standard_native_trace_wrapper_20261105_codex import as_list

    value = as_list(np.array([1.0, np.nan, np.inf, -np.inf]))
    assert value == [1.0, None, None, None]
    json.dumps(value, allow_nan=False)
