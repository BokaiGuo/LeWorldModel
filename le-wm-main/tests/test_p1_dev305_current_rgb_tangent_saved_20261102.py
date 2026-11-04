import numpy as np

from experiments.spectral_reachability import p1_dev305_current_rgb_tangent_saved_20261102 as saved


def test_current_metric_is_positive_definite_and_goal_guard_remains_closed():
    spec = saved.source.protocol()
    assert spec["goal_visibility"]["eligible"] is False
    assert all(role["visibility"]["eligible"] for role in spec["roles"])
    report = saved.base.read(saved.ROOT / "results" / f"{saved.source.NAME}_RUN_DEV01/report.json", saved.REPORT_SHA)
    for row in report["rows"]:
        sqrt = np.asarray(row["q_sqrt"])
        assert np.linalg.eigvalsh(sqrt).min() > 0
        assert np.isclose(np.trace(sqrt @ sqrt), 192.0)


def test_saved_selection_reconstruction_and_roster():
    result = saved.analyze()
    assert [r["arm"] for r in result["rows"]] == ["candidate_02", "candidate_00", "zero"]
    assert not result["p1_admission"]
    assert not result["full_cost_certified"]
