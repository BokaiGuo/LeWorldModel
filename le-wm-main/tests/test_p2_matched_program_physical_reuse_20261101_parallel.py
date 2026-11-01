from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p2_matched_program_physical_reuse_20261101_parallel as reuse


NEW_RUN = (
    reuse.ROOT / "results/p2_matched_program_host_20261101_parallel_RUN_02"
)
NEW_REPORT_SHA = (
    "3db63a0eadd898283222774431157676bea698f7754ca22f0329663fb013b749"
)
NEW_SAVED = (
    reuse.ROOT / "results/p2_matched_program_saved_20261101_parallel_RECEIPT_01.json"
)
NEW_SAVED_SHA = (
    "a364af8c5f632c8da564ef6c4389dc51764d76078124f788c881acb76323c6cd"
)


def test_real_saved_physical_lineage_is_action_identical():
    receipt = reuse.consume(
        NEW_RUN, NEW_REPORT_SHA, NEW_SAVED, NEW_SAVED_SHA
    )
    assert receipt["reused_physical_trajectories"] == 8
    assert receipt["new_physical_calls"] == 0
    assert receipt["native_successes"] == 0
    assert receipt["measured_resource_match"] is True
    assert receipt["action_identity_verified"] is True
    assert receipt["P2_admission"] is False
    assert len(receipt["pairs"]) == 4


def test_action_extractor_rejects_wrong_arm_and_length():
    branch = json.loads((NEW_RUN / "branch_0.json").read_bytes())
    damaged = deepcopy(branch)
    damaged["arm"] = "other"
    with pytest.raises(ValueError):
        reuse.winner_action_hex(damaged)
    damaged = deepcopy(branch)
    damaged["winner"]["raw_float32_hex"] = "00"
    with pytest.raises(ValueError):
        reuse.winner_action_hex(damaged)
