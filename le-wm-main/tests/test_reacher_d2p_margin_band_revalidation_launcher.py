from pathlib import Path


def test_p1_margin_band_revalidation_launcher_is_host_cuda_only_and_root_disjoint():
    launcher = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/run_reacher_d2p_p1_margin_band_revalidation_v2_cuda.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "nvidia-smi" in text
    assert "--device cuda" in text
    assert 'SEED_BASE="${3:-20269800}"' in text
    assert 'PREFLIGHT_SEED="${SEED_BASE}"' in text
    assert 'SOURCE_CALIBRATION_SEED="$((SEED_BASE + 100))"' in text
    assert 'BOUND_CALIBRATION_SEED="$((SEED_BASE + 200))"' in text
    assert 'OUTCOME_SEED="$((SEED_BASE + 300))"' in text
    assert '--seed "${PREFLIGHT_SEED}" --roots 4 --min-model-witness-roots 2' in text
    assert '--seed "${SOURCE_CALIBRATION_SEED}" --roots 16 --calibration-only' in text
    assert '--seed "${BOUND_CALIBRATION_SEED}" --roots 16 --calibration-only' in text
    assert '--seed "${OUTCOME_SEED}" --roots 24 --outcome-only' in text
    assert "--margin-band-selection-evidence \"${SELECTION}\"" in text
    assert "--calibration-evidence \"${BOUND_CALIBRATION}/report.json\"" in text
    assert "outcome roots remain sealed" in text
