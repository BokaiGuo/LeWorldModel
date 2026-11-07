import numpy as np
import pytest

from experiments.pc_rgb_cem_incumbent_selection_20261107_codex import choose


def batches():
    return [{"actions": np.zeros((1, n, 3, 10)).tolist(),
             "costs": [np.arange(n, dtype=float).tolist()]} for n in (32, 32, 2)]


def test_global_best_and_old_final_options():
    data = batches()
    data[0]["costs"][0][5] = -1
    data[2]["costs"][0] = [3, 2]
    _, _, best, old = choose(data)
    assert (best, old) == (5, 65)


def test_nonfinite_rejected():
    data = batches()
    data[0]["costs"][0][0] = float("nan")
    with pytest.raises(ValueError):
        choose(data)
