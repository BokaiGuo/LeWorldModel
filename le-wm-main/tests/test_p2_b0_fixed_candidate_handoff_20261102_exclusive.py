from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p2_b0_fixed_candidate_handoff_20261102_exclusive as m


def fixture():
    raw = np.zeros(20, dtype="<f4").tobytes().hex()
    return {"verified": {"status": "verified_saved_pair_known_root_not_admitted",
                         "report_sha256": m.isolation.REPORT_SHA,
                         "P1_admission": False, "P2_admission": False},
            "arms": [{"arm": arm, "candidates": [{"index": i, "raw_float32_hex": raw}
                                                  for i in range(16)]} for arm in m.ARMS]}


def test_fixed_census():
    result = m.validate(fixture())
    assert result == {"candidate_count_per_arm": 16, "same_full_action_at_same_index": 16,
                      "same_first_packet_at_same_index": 16}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nan", "source", "bounds"])
def test_fail_closed(kind):
    value = deepcopy(fixture())
    if kind == "missing":
        value["arms"][0]["candidates"].pop()
    elif kind == "duplicate":
        value["arms"][0]["candidates"][1]["index"] = 0
    elif kind == "nan":
        value["arms"][0]["candidates"][0]["raw_float32_hex"] = np.full(20, np.nan, dtype="<f4").tobytes().hex()
    elif kind == "source":
        value["verified"]["P2_admission"] = True
    else:
        value["arms"][0]["candidates"][0]["raw_float32_hex"] = np.full(20, 2, dtype="<f4").tobytes().hex()
    with pytest.raises(ValueError):
        m.validate(value)
