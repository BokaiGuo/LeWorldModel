import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_waypoint_cross_suffix_20261103_codex import (
    ROLES, SOURCE, run_record, source_arms,
)
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import replay


def test_all_cells_bind_the_saved_prefix_and_suffix():
    source = json.loads(SOURCE.read_text())["rows"][0]
    prefixes, suffixes = source_arms(source)
    result = run_record(source)
    assert len(result["cells"]) == 9
    assert {(cell["prefix_role"], cell["suffix_role"]) for cell in result["cells"]} == {
        (prefix, suffix) for prefix in ROLES for suffix in ROLES
    }
    for cell in result["cells"]:
        prefix, suffix = cell["prefix_role"], cell["suffix_role"]
        actions = np.asarray(cell["replay"]["actions"])
        np.testing.assert_array_equal(actions[:4], prefixes[prefix])
        np.testing.assert_array_equal(actions[4:], suffixes[suffix])
        assert replay(actions) == cell["replay"]
        assert cell["distance_change_vs_zero_tail"] == pytest.approx(
            cell["replay"]["terminal_error"] - result["zero_tail"][prefix]["terminal_error"]
        )


def test_tampered_saved_actions_fail_closed():
    source = json.loads(SOURCE.read_text())["rows"][0]
    bad = copy.deepcopy(source)
    bad["arms"]["selected"]["replay"]["actions"][0][0] = 0.
    with pytest.raises(ValueError, match="prefix/action mismatch"):
        source_arms(bad)
    bad = copy.deepcopy(source)
    bad["arms"]["zero"]["replay"]["terminal_error"] += 1
    with pytest.raises(ValueError, match="replay mismatch"):
        source_arms(bad)
