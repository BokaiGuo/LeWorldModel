import json

import numpy as np
import pytest

from experiments.spectral_reachability.pc_wall_triprefix_cem_intervention_20261103_codex import (
    SOURCE, evaluate_common, get_prefixes, optimize,
)
from experiments.spectral_reachability.pc_wall_unicycle_atlas_cem_20261103 import HIGH, LOW, replay


def test_prefix_binding_rejects_bad_source():
    report = json.loads(SOURCE.read_text())
    prefixes = get_prefixes(report)
    assert set(prefixes) == {"selected", "zero", "progress"}
    row = next(row for row in report["rows"] if row["seed"] == 20261103 and row["arm"] == "spectral")
    row["selected_root"] = "zero"
    with pytest.raises(ValueError, match="does not bind"):
        get_prefixes(report)


def test_optimized_suffix_replays_and_is_deterministic():
    prefixes = get_prefixes(json.loads(SOURCE.read_text()))
    a = optimize(prefixes["selected"], 20261103)
    b = optimize(prefixes["selected"], 20261103)
    assert a["suffix"] == b["suffix"]
    assert a["history"] == b["history"]
    assert a["suffix_backend"]["transitions"] == 3072
    assert a["suffix_backend"]["rollouts"] == 256
    assert a["prefix_backend"]["transitions"] == 4
    suffix = np.asarray(a["suffix"])
    assert np.all(suffix >= LOW) and np.all(suffix <= HIGH)
    checked = replay(np.concatenate([np.asarray(prefixes["selected"]), suffix]))
    assert checked == a["replay"]
    common = evaluate_common(prefixes, suffix)
    for role, item in common.items():
        assert item["distance_change_vs_zero_tail"] == pytest.approx(
            item["common_tail"]["terminal_error"] - item["zero_tail"]["terminal_error"]
        )
        np.testing.assert_array_equal(
            np.asarray(item["common_tail"]["actions"])[:4], prefixes[role]
        )
        np.testing.assert_array_equal(
            np.asarray(item["common_tail"]["actions"])[4:], suffix
        )
