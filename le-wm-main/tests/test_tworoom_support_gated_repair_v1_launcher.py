from __future__ import annotations

from pathlib import Path


def test_support_gated_launcher_uses_fresh_disjoint_gpu_panels() -> None:
    text = (Path(__file__).resolve().parents[1] / "experiments/gdd_wm/run_tworoom_support_gated_repair_v1_cuda.sh").read_text(encoding="utf-8")
    assert "SEEDS=(20264896 20265056 20265216)" in text
    assert "--support-gated-repair" in text
    assert "--roots 160" in text
    assert "gpu_preflight --device cuda --recurrent-smoke" in text
