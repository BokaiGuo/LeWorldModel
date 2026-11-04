import json

import numpy as np
import pytest

from experiments import pc_joint_carry_all_roots_saved_audit_20261103 as audit


def test_saved_bindings_and_tampered_score_rejected():
    report, source = (json.loads(p.read_text()) for p in (audit.run.OUTPUT, audit.run.SOURCE))
    with np.load(audit.run.SOURCE.with_suffix(".npz"), allow_pickle=False) as original, \
            np.load(audit.run.ARRAYS, allow_pickle=False) as arrays:
        entries = audit.run.roster(source, original)
        assert audit.verify(report, entries, arrays)["trajectories"] == 51
        report["rows"][1]["prior_score"] += 1
        with pytest.raises(ValueError, match="scores"):
            audit.verify(report, entries, arrays)
