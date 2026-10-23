from __future__ import annotations

from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "decision_benchmark"
    / "run_pointmaze_temporal_ood_v6_9_cuda.sh"
)


def test_v6_9_launcher_uses_module_entrypoints_for_every_python_stage() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'BUILDER_MODULE="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"' in text
    assert 'RUNNER_MODULE="experiments.decision_benchmark.run_pointmaze_temporal_ood_v6_interaction"' in text
    assert '"${PYTHON}" -m "${BUILDER_MODULE}"' in text
    assert 'experiments/run_with_torch_cudnn.sh "${PYTHON}" -m "${RUNNER_MODULE}" calibrate' in text
    assert 'experiments/run_with_torch_cudnn.sh "${PYTHON}" -m "${RUNNER_MODULE}" score' in text
    assert 'build_pointmaze_temporal_ood_v6_matched_panel.py' not in text
    assert 'run_pointmaze_temporal_ood_v6_interaction.py' not in text
