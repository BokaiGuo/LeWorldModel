import json
import numpy as np

from experiments.spectral_reachability import p1_diverse_adam_physical_20261030_main as physical


def test_new_suffixes_replace_only_prespecified_slots():
    joined = physical.source()
    old = json.loads(physical.previous.JOIN.read_bytes())
    plan = json.loads(physical.PLAN.read_bytes())
    assert joined["replaced_rows"] == [5, 13, 15]
    assert "not newly planned suffixes" in joined["replay_scope"]
    for i, row in enumerate(joined["rows"]):
        if i not in (5, 13, 15):
            assert row == old["rows"][i]
    for i, candidate in zip((5, 13, 15), plan["rows"], strict=True):
        assert joined["rows"][i]["raw15_float32_hex"][:80] == old["rows"][i]["raw15_float32_hex"][:80]
        assert joined["rows"][i]["raw15_float32_hex"][80:] == np.asarray(
            candidate["selected_actions"], dtype="<f4").tobytes().hex()


def test_saved_join_reconstructs_exactly():
    assert json.loads(physical.JOIN.read_bytes()) == physical.source()
