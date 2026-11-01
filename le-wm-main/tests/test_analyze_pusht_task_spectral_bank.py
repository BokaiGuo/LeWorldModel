from pathlib import Path
import pytest
from experiments.spectral_reachability.analyze_pusht_task_spectral_bank import analyze


def test_local_calibration_decomposition():
    directory=Path('results/pusht_task_spectral_bank4_v1_20261026_cuda')
    if not (directory/'report.json').exists():
        pytest.skip('local calibration not installed')
    result=analyze(directory)
    assert len(result['rows'])==4
    assert result['P1_admission'] is False
    assert result['model_inference'] is False
    for row in result['rows']:
        assert len(row['details'])==10
        assert row['reach_reduction']==pytest.approx(
            row['nominal_goal_residual_component']+row['incremental_linear_action_component'],abs=1e-12)
        for detail in row['details']:
            assert detail['linear_action_improvement'] >= -1e-12


def test_missing_report_not_silently_accepted(tmp_path):
    with pytest.raises(FileNotFoundError):analyze(tmp_path)
