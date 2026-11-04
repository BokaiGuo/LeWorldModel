import json

import numpy as np
import pytest

from experiments import pc_joint_carry_saved_audit_20261103 as audit


def test_saved_receipts_and_tampered_tail():
    report, source, stats = (json.loads(p.read_text()) for p in (audit.RESULT, audit.SOURCE, audit.STATS))
    with np.load(audit.RESULT.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(audit.SOURCE.with_suffix(".npz"), allow_pickle=False) as original:
        assert audit.verify(report, source, stats, data, original)["branches"] == 9
        changed = {key: data[key] for key in data.files}
        changed["zero_carry_init_actions"] = changed["zero_carry_init_actions"].copy()
        changed["zero_carry_init_actions"][7, 0] += np.float32(0.01)
        with pytest.raises(ValueError, match="sequence binding"):
            audit.verify(report, source, stats, changed, original)


def test_tampered_score_rejected():
    report, source, stats = (json.loads(p.read_text()) for p in (audit.RESULT, audit.SOURCE, audit.STATS))
    report["rows"][1]["prior_score"] += 1
    with np.load(audit.RESULT.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(audit.SOURCE.with_suffix(".npz"), allow_pickle=False) as original:
        with pytest.raises(ValueError, match="scores"):
            audit.verify(report, source, stats, data, original)
