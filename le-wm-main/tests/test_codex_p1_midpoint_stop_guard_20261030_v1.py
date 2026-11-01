import pytest

from experiments.spectral_reachability.codex_p1_midpoint_stop_guard_20261030_v1 import (
    require_replan_allowed,
)


def events():
    return [{"step": i, "terminated": False, "truncated": False} for i in range(1, 6)]


def test_accepts_complete_live_prefix():
    require_replan_allowed(events())


@pytest.mark.parametrize("field", ["terminated", "truncated"])
@pytest.mark.parametrize("step", [0, 4])
def test_rejects_stop_at_any_prefix_step(field, step):
    rows = events()
    rows[step][field] = True
    with pytest.raises(ValueError, match="native stop"):
        require_replan_allowed(rows)


def test_rejects_missing_or_non_boolean_flags_and_wrong_sequence():
    rows = events()
    del rows[2]["terminated"]
    with pytest.raises(ValueError, match="stop flag"):
        require_replan_allowed(rows)
    rows = events()
    rows[2]["truncated"] = 0
    with pytest.raises(ValueError, match="stop flag"):
        require_replan_allowed(rows)
    rows = events()
    rows[2]["step"] = 4
    with pytest.raises(ValueError, match="identity"):
        require_replan_allowed(rows)
    with pytest.raises(ValueError, match="incomplete"):
        require_replan_allowed(events()[:4])
