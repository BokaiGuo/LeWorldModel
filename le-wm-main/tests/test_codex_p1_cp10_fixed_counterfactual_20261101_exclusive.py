from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_cp10_fixed_counterfactual_20261101_exclusive as m
from experiments.spectral_reachability import codex_p1_cp10_fixed_counterfactual_verify_20261101_exclusive as v


def test_schedule_pairs_same_first_ten_but_uses_both_last_packets():
    entries = {}
    for i, role in enumerate(m.ROLES):
        entries[role] = {"first10": np.full((10, 2), 0.1 * i, np.float32).tolist(),
                         "hold_last": np.zeros((5, 2), np.float32).tolist(),
                         "optimized_last": np.full((5, 2), 0.2, np.float32).tolist(),
                         "cp10": {k: role for k in m.IDENTITY}}
    slots = m.schedule(entries)
    assert len(slots) == 6
    for i in range(0, 6, 2):
        a, b = slots[i:i+2]
        assert a["role"] == b["role"]
        assert (a["tail"], b["tail"]) == ("hold", "optimized")
        assert a["commands"][:10] == b["commands"][:10]
        assert a["commands"][10:] != b["commands"][10:]


def test_cp10_guard_requires_all_identity_and_exact_actions():
    identity = {k: [k] for k in m.IDENTITY}
    commands = np.zeros((10, 2)).tolist()
    m.check_cp10(identity, identity, commands, commands)
    bad = deepcopy(identity)
    bad["pixel_sha256"] = "different"
    with pytest.raises(Exception):
        m.check_cp10(identity, bad, commands, commands)
    bad_commands = deepcopy(commands)
    bad_commands[9][0] = 0.25
    with pytest.raises(Exception):
        m.check_cp10(identity, identity, commands, bad_commands)


def test_bad_action_or_unknown_roster_fails_closed():
    with pytest.raises(Exception):
        m.schedule({})
    with pytest.raises(Exception):
        m.raw_actions([[float("nan"), 0]], 1)


def test_cp10_identity_seal_required_before_suffix_and_native_stop():
    events = []
    for i in range(1, 16):
        events.append({"kind": "branch_dispatch", "step": i, "index": len(events), "command": [0, 0]})
        events.append({"kind": "branch_returned", "step": i, "index": len(events), "command": [0, 0],
                       "terminated": False, "truncated": False})
        if i == 10:
            events.append({"kind": "cp10_identity_sealed", "index": len(events)})
    v.check_order(events, 15)
    with pytest.raises(ValueError):
        v.check_order([e for e in events if e["kind"] != "cp10_identity_sealed"], 15)
    broken = deepcopy(events)
    next(e for e in broken if e["kind"] == "branch_returned" and e["step"] == 5)["terminated"] = True
    with pytest.raises(ValueError):
        v.check_order(broken, 15)
    v.check_order(events[:14], 7)
