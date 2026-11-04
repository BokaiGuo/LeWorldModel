"""Inventory uses saved arrays only, never loads or calls a neural model."""
import pytest

from experiments.spectral_reachability import pc_pointmaze_velocity_score_inventory_20261103_codexexclusive as audit


def test_real_saved_inventory_full_roster_and_missing_cells():
    result = audit.inventory()
    assert result["fixed_candidates"] == 384
    assert result["available_visual_cells"] == result["missing_proprio_cells"] == 1152
    assert result["velocity_roots"] == [1884, 1885, 1886, 1887]
    assert result["separate_score_roots"] == [1889, 1893, 1895]
    assert result["cross_table_complete"] is False
    assert result["can_join_separate_score_bank"] is False
    assert not result["neural_inference"] and not result["physical_execution"]


def test_refuse_any_source_hash_drift(monkeypatch):
    inputs = dict(audit.INPUTS)
    name, _ = inputs["velocity_json"]
    inputs["velocity_json"] = (name, "0" * 64)
    monkeypatch.setattr(audit, "INPUTS", inputs)
    with pytest.raises(ValueError, match="SHA mismatch"):
        audit.inventory()
