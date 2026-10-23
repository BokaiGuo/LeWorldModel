from __future__ import annotations

from pathlib import Path


def test_v2_launcher_uses_a_fresh_bound_seed_protocol() -> None:
    path = Path(__file__).resolve().parents[1] / "experiments/gdd_wm/run_tworoom_verified_local_calibrated_v2_cuda.sh"
    text = path.read_text(encoding="utf-8")
    assert 'PROTOCOL_ID="tworoom_verified_local_calibrated_v2_20261023"' in text
    assert 'CAL_SEED=20264224' in text
    assert 'OUTCOME_SEED=20264288' in text
    assert 'CALIBRATION_EPISODES=64' in text
    assert 'OUTCOME_EPISODES=128' in text
    assert '--expected-seed-start "${CAL_SEED}" --expected-root-count "${CALIBRATION_EPISODES}"' in text
    assert '--expected-seed-start "${OUTCOME_SEED}" --expected-root-count "${OUTCOME_EPISODES}"' in text
    assert "gdd_wm_tworoom_verified_local_calibrated_v1_20261023_cuda" not in text
