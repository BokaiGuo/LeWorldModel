from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_pointmaze_htsc_passive_local_v9_host_cuda.sh"
)


def test_v9_passive_launcher_uses_fresh_roots_and_host_cuda_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "point_maze_v35_passive_local_extension_20261022" in text
    assert "ROOT_MIN=13544" in text and "ROOT_MAX_EXCLUSIVE=13752" in text
    assert "VISUAL_ROOTS=\"1928" in text
    assert "CONSISTENCY_ROOTS=\"1944" in text
    assert "--device cuda --recurrent-smoke" in text
    assert "experiments.run_pointmaze_htsc_passive_local_strict" in text
    assert "refusing to reuse immutable passive-local output path" in text
    assert "Codex bwrap user namespace cannot access /dev/nvidia" in text
