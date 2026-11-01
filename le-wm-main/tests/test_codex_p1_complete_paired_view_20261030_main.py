import copy

import pytest

from experiments.spectral_reachability import codex_p1_complete_paired_view_20261030_main as view


@pytest.fixture
def sources():
    return (view.gap.read(view.TABLE, view.TABLE_SHA)[0],
            view.gap.read(view.JOIN, view.JOIN_SHA)[0],
            view.gap.read(view.gap.PAIRS, view.gap.PAIR_SHA)[0])


def test_full_roster_stops_and_no_duplicate_independent_observations(sources):
    result = view.build(*sources)
    assert len(result["rows"]) == 24
    probes = [p for r in result["rows"] for p in r["common_suffixes"]]
    assert len(probes) == 1008
    stopped = [p for p in probes if p["status"] == "native_stopped"]
    assert len(stopped) == 2
    assert all(p["common_suffix_endpoint_native_metrics"] is None for p in stopped)
    assert all(len(r["independent_suffixes"]) == 2 for r in result["rows"])
    assert result["equal_total_budget_certified"] is False
    assert result["P1_admission"] is False


def test_reject_action_and_prefix_binding_changes(sources):
    table, joined, planned = sources
    corrupted = copy.deepcopy(table)
    corrupted["rows"][0]["probe_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="common action"):
        view.build(corrupted, joined, planned)
    corrupted_plan = copy.deepcopy(planned)
    corrupted_plan["rows"][0]["prefix5_float32_hex"] = "0" * 80
    with pytest.raises(ValueError, match="anchor identity"):
        view.build(table, joined, corrupted_plan)


def test_numeric_null_and_zero_baseline():
    assert view.delta(None, {}) is None
    values = dict.fromkeys(view.FIELDS, 3.0)
    assert all(x == 0 for x in view.delta(values, values).values())
