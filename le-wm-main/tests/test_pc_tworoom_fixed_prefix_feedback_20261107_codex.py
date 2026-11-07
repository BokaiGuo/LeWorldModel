import numpy as np
import pytest

from experiments.pc_tworoom_fixed_prefix_feedback_20261107_codex import check_same_prefix


def test_common_prefix_matches_sealed_actions():
    chain = np.arange(50, dtype=np.float32).reshape(25, 2) / 100
    trace = [{"action": action.tolist()} for action in chain[:5]]
    check_same_prefix(chain, trace)
    trace[2]["action"][0] = -0.9
    with pytest.raises(ValueError, match="mismatch"):
        check_same_prefix(chain, trace)


def test_reject_incomplete_prefix():
    with pytest.raises(ValueError, match="incomplete"):
        check_same_prefix(np.zeros((25, 2)), [])
