from experiments.spectral_reachability import p1_dev304_diverse_feedback_20261102 as feedback
from experiments.spectral_reachability import p1_dev304_diverse_feedback_saved_20261102 as saved


def test_feedback_contract_is_rgb_action_only():
    contract = feedback.protocol()
    assert len(contract["roles"]) == 2
    assert all(set(role) == {"arm", "images", "actual_history", "prefix_sha256",
                             "contact_direction_raw_xy"} for role in contract["roles"])
    assert not contract["independent_confirmation"]


def test_recorded_feedback_execution():
    result = saved.analyze()
    assert result["frames_verified"] == 18
    assert all(not row["native_success"] for row in result["rows"])
    assert not result["p1_admission"]
