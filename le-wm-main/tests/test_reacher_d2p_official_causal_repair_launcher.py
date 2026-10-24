from pathlib import Path


def test_official_repair_is_cuda_only_and_has_fresh_fail_closed_gates():
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    acquire = (root / "acquire_reacher_official_dataset.sh").read_text(encoding="utf-8")
    launcher = (root / "run_reacher_d2p_official_causal_repair_cuda.sh").read_text(encoding="utf-8")
    assert "HF_ENDPOINT=https://hf-mirror.com" in acquire
    assert "quentinll/lewm-reacher" in acquire
    assert "official_source_manifest.json" in acquire
    assert "--zstd" in acquire
    assert "--device cuda" in launcher
    assert "CPU fallback" in launcher
    assert "official_source_manifest.json" in launcher
    assert "--outcome-only" in launcher
    assert "observed_encoded_goal_cost" in launcher
    assert "P0 did not unlock P1" in launcher
    assert "outcome roots remain sealed" in launcher


def test_after_acquisition_launcher_waits_for_validation_and_execs_cuda_pipeline():
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    launcher = (root / "run_reacher_d2p_official_causal_repair_when_ready_cuda.sh").read_text(encoding="utf-8")
    assert "official_source_manifest.json" in launcher
    assert "systemctl --user is-active --quiet" in launcher
    assert "sleep 30" in launcher
    assert "exec bash" in launcher
    assert "run_reacher_d2p_official_causal_repair_cuda.sh" in launcher
