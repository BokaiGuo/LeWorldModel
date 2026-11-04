"""CPU-only join tests; synthetic fixtures are not physical evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p3_venv_winner_physical_prepare_20261102_codexexclusive as m


def fixture():
    slots = [{"source_slot": r, "root_sha256": f"root{r}", "task_id": f"task{r}",
              "task_sha256": f"hash{r}"} for r in range(2)]
    refs = [{"root_sha256": f"root{r}"} for r in range(2)]
    winners = [{"index": i, "cost": float(i), "commands": [[0.1, 0.2]] * 10} for i in range(2)]
    checked = {"rows": [{"arm": "signal", "allocation": [15, 5], "winners": deepcopy(winners)},
                        {"arm": "reverse", "allocation": [5, 15], "winners": deepcopy(winners)}]}
    return checked, slots, refs


def bridge(slot, ref, bindings):
    bindings[slot["task_id"]] = slot["task_sha256"]


def test_join_has_four_ordered_raw_bound_plans():
    checked, slots, refs = fixture()
    plans, bindings = m.join(checked, slots, refs, bridge, lambda x: np.asarray(x, dtype=np.float32))
    assert [(x["arm"], x["root_slot"]) for x in plans] == [
        ("signal", 0), ("signal", 1), ("reverse", 0), ("reverse", 1)]
    assert len(bindings) == 2 and all(len(p["raw_float32_sha256"]) == 64 for p in plans)


def test_wrong_root_or_action_shape_is_rejected():
    checked, slots, refs = fixture()
    refs[1]["root_sha256"] = "wrong"
    with pytest.raises(ValueError, match="root binding"):
        m.join(checked, slots, refs, bridge, lambda x: np.asarray(x, dtype=np.float32))
    checked, slots, refs = fixture()
    checked["rows"][0]["winners"][0]["commands"] = [[0.1, 0.2]]
    with pytest.raises(ValueError, match="action chart"):
        m.join(checked, slots, refs, bridge, lambda x: np.asarray(x, dtype=np.float32))


def test_reordered_arms_rejected():
    checked, slots, refs = fixture()
    checked["rows"].reverse()
    with pytest.raises(ValueError, match="roster"):
        m.join(checked, slots, refs, bridge, lambda x: np.asarray(x, dtype=np.float32))
