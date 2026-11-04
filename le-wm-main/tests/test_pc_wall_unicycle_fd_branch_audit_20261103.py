import json

import numpy as np

from experiments.spectral_reachability import pc_wall_unicycle_fd_branch_audit_20261103 as audit


def test_trace_detects_wall_transition_discontinuity():
    backend = audit.TracingBackend()
    context = dict(audit.wall.ROOT)
    context["latent_history"] = [[-.1, 0., 0.], [-.1, 0., 0.]]
    jac = backend.jacobian(context, np.array([[.1, 0.]]))
    assert backend.fd_audit[0]["branch_different"] == 1
    assert backend.fd_audit[0]["different_coordinates"] == [0]
    assert abs(jac[0, 0]) > 1000


def test_saved_trace_totals():
    result_path = audit.ROOT / "results/pc_wall_unicycle_fd_branch_audit_20261103_DEV01.json"
    report = json.loads(result_path.read_text())
    for row in report["rows"]:
        assert row["jacobians"] == len(row["details"]) == 31
        assert row["coordinate_pairs"] == sum(item["pair_count"] for item in row["details"]) == 248
        assert row["differing_pairs"] == sum(item["branch_different"] for item in row["details"])
