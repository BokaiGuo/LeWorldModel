import json

import numpy as np
import pytest

from experiments import pc_joint_carry_all_roots_20261103 as exp


def test_roster_covers_every_saved_branch_and_binds_plans():
    source = json.loads(exp.SOURCE.read_text())
    with np.load(exp.SOURCE.with_suffix(".npz"), allow_pickle=False) as arrays:
        entries = exp.roster(source, arrays)
    assert len(entries) == 17
    assert [sum(e["root"] == r for e in entries) for r in exp.ROOTS] == [4, 4, 5, 4]
    assert [(e["root"], e["role"]) for e in entries if e["role"].endswith("qualified")] == [(8349, "raw_qualified")]


def test_unknown_or_absent_decision_is_rejected():
    source = json.loads(exp.SOURCE.read_text())
    with pytest.raises(ValueError):
        exp.indices(source["rows"][0], "raw_qualified")
    with pytest.raises(ValueError):
        exp.indices(source["rows"][0], "unrecognized")
