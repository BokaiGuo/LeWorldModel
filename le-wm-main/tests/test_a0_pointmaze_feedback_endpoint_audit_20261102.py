import json
from pathlib import Path

import numpy as np

from experiments import a0_pointmaze_feedback_endpoint_audit_20261102 as audit


def test_saved_audit_binds_fixed_source_and_all_methods():
    report = json.loads(audit.OUTPUT.read_text())
    source = json.loads(audit.SOURCE.read_text())
    assert report["feedback_result_sha256"] == audit.digest(audit.SOURCE) == audit.SOURCE_SHA
    assert report["code_sha256"] == audit.digest(Path(audit.__file__))
    assert len(report["rows"]) == 8
    expected = {(row["root"], b["method"], b["first"], b["chosen_tail"])
                for row in source["rows"] for b in row["branches"]}
    actual = {(row["root"], row["method"], row["first"], row["tail"])
              for row in report["rows"]}
    assert actual == expected
    for row in report["rows"]:
        old = np.asarray(row["legacy_endpoint_eval_only"])
        new = np.asarray(row["position_backend_endpoint_eval_only"])
        assert np.isclose(np.linalg.norm(old[:2] - new[:2]), row["position_delta_l2"])
        assert np.isclose(np.linalg.norm(old[2:] - new[2:]), row["velocity_delta_l2"])
        assert row["rgb5_matches_source"] and row["rgb10_matches_source"]
