from __future__ import annotations

from pathlib import Path


def test_mechanics_v4_launcher_uses_fresh_seed_bands_and_forensic_audit() -> None:
    text = (Path(__file__).resolve().parents[1] / "experiments/run_pusht_action_conditioned_mechanics_v4_cuda.sh").read_text(encoding="utf-8")
    assert "1704240000" in text and "1704250000" in text and "1704260000" in text
    assert "--confirmation-representation geometry_plus_pair_mechanics" in text
    assert "gpu_preflight --device cuda --recurrent-smoke" in text
    assert "audit_pusht_action_conditioned_signed_torque_confirmation" in text
