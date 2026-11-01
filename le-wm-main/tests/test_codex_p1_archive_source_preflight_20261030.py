import copy

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_archive_source_preflight_20261030 as m


class Array:
    def __init__(self, x, dtype):
        self.x = np.asarray(x, dtype=dtype)
        self.shape, self.dtype = self.x.shape, self.x.dtype

    def __getitem__(self, key):
        return self.x[key]


def fixture():
    f = {
        "ep_len": Array([2, 2, 2], "int32"),
        "ep_offset": Array([0, 2, 4], "int64"),
        "episode_idx": Array([0, 0, 1, 1, 2, 2], "int64"),
        "step_idx": Array([0, 1, 0, 1, 0, 1], "int64"),
        "state": Array([[0] * 7, [99] * 7, [0] * 7, [99] * 7,
                        [5, 0, 0, 0, 0, 0, 0], [99] * 7], "float32"),
    }
    schema = {key: (a.shape, str(a.dtype)) for key, a in f.items()}
    history = {"observed_prefix_records": [{"physical_pose": [0] * 5}] * 80,
               "history_complete": False}
    return f, history, schema


def test_all_fixed_rows_preserved_and_collisions_symmetric():
    f, history, schema = fixture()
    rows = m.scan(f, history, (0, 1, 2), schema)
    assert [x["episode_ordinal"] for x in rows] == [0, 1, 2]
    assert rows[0]["batch_pose_matches"] == [1]
    assert rows[1]["batch_pose_matches"] == [0]
    assert len(rows[0]["known_prefix_pose_matches"]) == 80
    assert rows[2]["known_prefix_pose_matches"] == []


@pytest.mark.parametrize("case", ["extent", "step", "episode", "nan", "schema", "duplicate", "history"])
def test_fail_closed(case):
    f, h, schema = fixture()
    if case == "extent":
        f["ep_offset"].x[1] = 3
    elif case == "step":
        f["step_idx"].x[0] = 1
    elif case == "episode":
        f["episode_idx"].x[0] = 2
    elif case == "nan":
        f["state"].x[0, 0] = np.nan
    elif case == "schema":
        schema = copy.deepcopy(schema)
        schema["state"] = ((6, 7), "float64")
    elif case == "history":
        h["history_complete"] = True
    with pytest.raises(ValueError):
        m.scan(f, h, (0, 0) if case == "duplicate" else (0, 1), schema)


def test_no_access_to_outcome_or_future_rows():
    f, h, schema = fixture()
    class Guard:
        def __init__(self, inner):
            self.inner, self.shape, self.dtype = inner, inner.shape, inner.dtype
        def __getitem__(self, index):
            assert index in (0, 2)
            return self.inner[index]
    f["state"] = Guard(f["state"])
    assert len(m.scan(f, h, (0, 1), schema)) == 2
