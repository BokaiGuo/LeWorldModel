import copy

import pytest

from experiments.spectral_reachability import codex_p1_live_twoplan_pair_20261030_main as pair


@pytest.fixture(scope="module")
def inputs():
    return (pair.read(pair.VIEW, pair.VIEW_SHA), pair.read(pair.PREP, pair.PREP_SHA),
            pair.read(pair.SAVED, pair.SAVED_SHA))


def test_all_rows_and_censor(inputs):
    rows = pair.join(*inputs)
    assert len(rows) == 24
    assert all(len(row["common_suffixes"]) == 42 for row in rows)
    stopped = [r for r in rows if r["live_twoplan"]["status"] == "native_stopped"]
    assert len(stopped) == 2
    assert all(r["live_twoplan"]["endpoint_metrics"] is None
               and r["live_twoplan"]["endpoint_minus_own_zero_suffix"] is None for r in stopped)
    assert all(r["live_twoplan"]["last_observed_metrics"]["native_success"] for r in stopped)


@pytest.mark.parametrize("field", ["task_id", "physical_root_sha256", "prefix5_float32_hex"])
def test_rejects_identity_mismatch(inputs, field):
    view, prep, saved = copy.deepcopy(inputs)
    prep["roster"][0][field] = "tampered"
    with pytest.raises(ValueError):
        pair.join(view, prep, saved)


def test_rejects_checkpoint_mismatch(inputs):
    view, prep, saved = copy.deepcopy(inputs)
    prep["roster"][0]["checkpoint"]["snapshot"]["state"][0] += 1
    with pytest.raises(ValueError):
        pair.join(view, prep, saved)


def test_rejects_censor_mismatch(inputs):
    view, prep, saved = copy.deepcopy(inputs)
    stopped = next(row for row in saved["rows"] if row["status"] == "native_stopped")
    stopped["endpoint_metrics"] = stopped["last_observed_metrics"]
    with pytest.raises(ValueError):
        pair.join(view, prep, saved)


def test_rejects_missing_probe(inputs):
    view, prep, saved = copy.deepcopy(inputs)
    view["rows"][0]["common_suffixes"].pop()
    with pytest.raises(ValueError):
        pair.join(view, prep, saved)
