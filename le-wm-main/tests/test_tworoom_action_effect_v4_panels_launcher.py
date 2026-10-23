from __future__ import annotations

from pathlib import Path


def test_v4_launcher_declares_three_disjoint_160_root_panels() -> None:
    text = (Path(__file__).resolve().parents[1] / "experiments/gdd_wm/run_tworoom_action_effect_v4_panels_cuda.sh").read_text(encoding="utf-8")
    assert "SEEDS=(20264416 20264576 20264736)" in text
    assert "--roots 160" in text
    assert "gpu_preflight --device cuda --recurrent-smoke" in text
    assert "audit_tworoom_action_effect_v4_panels" in text
