import copy

import pytest

from experiments.spectral_reachability import p1_response_planned_pair_20261031 as analysis


def load_sources():
    table = analysis.table
    read = table.panel.physical.bundle.physical.prior.native.read
    return [read(path, cap=64 * 1024**2) for path in (
        analysis.PREP, table.control.OLD, table.control.PAIR, table.PLANNED,
        table.NEW_PLAN, table.OLD_PLAN)]


def test_real_saved_pairing_preserves_censor_and_control_identity():
    rows = analysis.analyze(*load_sources())
    assert len(rows["rows"]) == 14
    assert rows["fixed_prefixes"] == 16
    assert rows["censored_before_cp5"] == [1, 5]
    censored = next(row for row in rows["rows"] if row["status"] == "censored_out_of_domain")
    assert censored["endpoint_metrics"] is None
    assert all(control["endpoint_minus_control"] is None for control in censored["controls"].values())
    assert rows["equal_budget_certified"] is False
    assert rows["P1_admission"] is False


def test_tampered_endpoint_and_initializer_are_rejected():
    sources = load_sources()
    corrupt = copy.deepcopy(sources)
    censored = next(row for row in corrupt[3]["rows"] if row["status"] == "censored_out_of_domain")
    censored["completed_endpoint"] = [0] * 7
    with pytest.raises(Exception, match="censored endpoint"):
        analysis.analyze(*corrupt)
    sources[5]["rows"][1]["initial_f32_sha256"] = "invalid"
    with pytest.raises(Exception, match="matched initialization"):
        analysis.analyze(*sources)
