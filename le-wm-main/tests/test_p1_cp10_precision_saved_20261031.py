import copy
import json

import pytest

from experiments.spectral_reachability import p1_cp10_precision_saved_20261031 as saved


def test_saved_precision_and_tampering():
    result = json.loads(saved.RESULT.read_text())
    reference = json.loads(saved.experiment.REFERENCE.read_text())
    rows = saved.verify(result, reference)
    assert len(rows) == 2
    assert rows[1]["fp32"] < 0 and rows[1]["bf16"] < 0 and rows[1]["real"] > 0
    broken = copy.deepcopy(result)
    broken["rows"][1]["modes"]["bf16"]["feedback"]["prediction"][0] += 10
    with pytest.raises(ValueError):
        saved.verify(broken, reference)
    broken = copy.deepcopy(result)
    broken["rows"][0]["modes"]["bf16"]["open"]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        saved.verify(broken, reference)
