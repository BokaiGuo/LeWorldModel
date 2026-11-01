import copy

import pytest

from experiments.spectral_reachability import p1_dev303_dynamics_physical_20261102_main as p


def report():
    return {"status": "model_only_complete", "P1_admission": False,
            "selections": {"latent": 0, "goal_tangent": 0},
            "rows": [{"index": 0, "label": "spectral", "prefix": [[0., 0.]] * 5,
                      "metrics": {name: {"tail": [[0., 0.]] * 10}
                                  for name in ("latent", "goal_tangent")}}]}


def test_all_metric_actions_preserved_without_outcomes():
    value = report()
    before = copy.deepcopy(value)
    specs = p.specs_from_report(value)
    assert len(specs) == 2 and value == before
    assert [s["metric"] for s in specs] == ["latent", "goal_tangent"]
    assert all(s["selected_before_physics"] for s in specs)


def test_bad_actions_refused():
    value = report()
    value["rows"][0]["metrics"]["latent"]["tail"] = [[2., 0.]] * 10
    with pytest.raises(ValueError):
        p.specs_from_report(value)


def test_missing_candidate_refused():
    value = report()
    value["rows"][0]["index"] = 3
    with pytest.raises(ValueError):
        p.specs_from_report(value)
