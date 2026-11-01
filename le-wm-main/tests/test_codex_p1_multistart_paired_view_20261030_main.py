import copy

import pytest

from experiments.spectral_reachability import codex_p1_multistart_paired_view_20261030_main as view


@pytest.fixture
def sources():
    return [view.original.gap.read(path, sha)[0] for path, sha in zip(
        (view.VIEW, view.SAVED, view.PLAN), view.SHAS, strict=True)]


def test_roster_and_stops(sources):
    result = view.build(*sources)
    assert len(result["rows"]) == 24
    stopped = [r["multistart20_exploratory"] for r in result["rows"]
               if r["multistart20_exploratory"]["status"] == "native_stopped"]
    assert len(stopped) == 2
    assert all(r["horizon_metrics"] is None and r["endpoint_minus_own_zero_suffix"] is None
               for r in stopped)
    assert result["P1_admission"] is False


def test_reject_mismatched_identity_and_action(sources):
    bad = copy.deepcopy(sources)
    bad[1]["rows"][0]["condition"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        view.build(*bad)
    bad = copy.deepcopy(sources)
    bad[2]["rows"][0]["selected_action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action"):
        view.build(*bad)
