from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_residual_cp5_saved_20261102 as subject


def test_complete_real_saved_panel():
    result = subject.analyze()
    assert len(result["rows"]) == 8
    assert result["frame_references_verified"] == 144
    assert result["P1_admission"] is False
    assert result["independent_confirmation"] is False
    assert {row["metric"] for row in result["rows"]} == {"latent", "goal_tangent"}


@pytest.mark.parametrize("kind", ["action", "termination", "roster"])
def test_saved_trace_corruption_rejected(monkeypatch, kind):
    read = subject.physical.read

    def changed(path, expected=None):
        value = read(path, expected)
        if path == subject.RUN / "report.json":
            value = deepcopy(value)
            if kind == "roster":
                value["rows"].pop()
            elif kind == "action":
                value["rows"][0]["transitions"][-1]["action"][0] += .01
            else:
                value["rows"][0]["transitions"][-1]["terminated"] = True
        return value

    monkeypatch.setattr(subject.physical, "read", changed)
    with pytest.raises(ValueError):
        subject.analyze()
