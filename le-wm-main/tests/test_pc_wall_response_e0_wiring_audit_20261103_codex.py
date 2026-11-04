import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pc_wall_response_e0_wiring_audit_20261103_codex as audit


def test_parameters_roundtrip_and_rejects_non_template():
    p = np.array([.1, -.2, .3, -.4])
    actions = audit.joint.decode(p[None])[0]
    np.testing.assert_allclose(audit.parameters(actions), p, atol=1e-12)
    actions[4, 1] = .1
    with pytest.raises(ValueError):
        audit.parameters(actions)


def test_saved_injection_and_fd_identity():
    report = json.loads(audit.OUTPUT.read_text())
    source = json.loads(audit.experiment.OUTPUT.read_text())
    assert all(audit.base.digest(Path(path)) == digest for path, digest in report["dependencies"].items())
    assert len(report["rows"]) == 8
    for row, old in zip(report["rows"], source["rows"], strict=True):
        trace, _ = audit.base.rollout(old["task"]["start"],
                                      audit.joint.decode(np.asarray(row["witness_parameters"])[None]))
        score = np.linalg.norm(trace[0, -1, :2] - old["task"]["goal"])
        np.testing.assert_allclose(score, row["witness_distance"], atol=1e-12)
        for item in row["injection_comparisons"]:
            assert item["chosen_witness"] == (int(np.argmin([item["witness_distance"], *item["planner_distances"]])) == 0)
        for item in row["response_probes"]:
            assert item["fd_scales"]["0.01"]["absolute_difference"] < 1e-10
