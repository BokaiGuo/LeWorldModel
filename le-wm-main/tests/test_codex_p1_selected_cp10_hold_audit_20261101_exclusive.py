from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_selected_cp10_hold_audit_20261101_exclusive as audit


def test_identity_rejects_each_mismatch_and_incomplete_actions():
    point = {key: key for key in audit.IDENTITY}
    row = {"executed_raw": [[i, i] for i in range(10)], "observations": [point] * 11}
    assert all(audit.identity(row, row).values())
    for key in audit.IDENTITY:
        altered = deepcopy(row)
        altered["observations"][10][key] = "changed"
        assert not audit.identity(row, altered)[key]
    altered = deepcopy(row)
    altered["executed_raw"] = altered["executed_raw"][:9]
    assert not audit.identity(row, altered)["first10_actions"]


def test_joint_distance_uses_first_four_dimensions():
    assert audit.joint_xy([3, 4, 0, 0, 999], [0, 0, 0, 0, 0]) == 5
