from __future__ import annotations

import inspect

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import build_pointmaze_v21_c3_reliability_pool as pool
from experiments.decision_benchmark import run_pointmaze_c3_calibrated_reliability_v21_cuda as v21
from experiments.gdd_wm import preflight_pointmaze_c3_calibrated_reliability_v21 as preflight


def test_v21_uses_fresh_roots_and_the_frozen_v20_route() -> None:
    assert pool._expected_high()["new_root_id_range"] == [28040, 28487]
    assert pool._expected_direct()["new_root_id_range"] == [28488, 28871]
    assert v21.ROOT_RANGE == (28040, 28872)
    assert v21.POLICY["direction"] == "recurrent_when_high"
    assert v21.INVERSE_POLICY["direction"] == "recurrent_when_low"
    assert preflight.SCHEMA == panel.C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA


def test_v21_schema_retains_all_velocity_alias_contract_checks() -> None:
    source = inspect.getsource(panel.frozen_preflight_provenance)
    assert "C3_CALIBRATED_RELIABILITY_V21_PREFLIGHT_SCHEMA" in source
    # New C3 panels must retain the same static split and velocity-alias
    # evidence as V20; accepting the schema alone is insufficient.
    assert "exact_split_schemas" in source
    assert "velocity_alias_schemas" in source
    builder_source = inspect.getsource(panel.build_panel)
    assert "frozen preflight lacks its grouped split-root ledger" in builder_source


def test_v21_seals_calibration_gate_before_outcome_labels() -> None:
    source = inspect.getsource(v21.main)
    assert source.index("gate_path.write_text") < source.index("outcome_path.write_text")
    assert "refusing to reopen immutable outcomes" in source
    assert "V20_PANEL / \"temporal_ood_v6_frozen_selection.json\"" in inspect.getsource(v21._costs)
    costs_source = inspect.getsource(v21._costs)
    assert "history_source = str(frozen.get" in costs_source
    assert "expected_history_source=history_source" in costs_source
