from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_counterfactual_contrast_v7_cuda as driver


def test_v7_driver_keeps_the_counterfactual_protocol_bound_to_cuda() -> None:
    source = inspect.getsource(driver.main)
    assert driver.OUTPUT.name == "pointmaze_counterfactual_contrast_v7_20261022_cuda"
    assert len(driver.CONTEXTS) >= 12
    assert all(path.is_absolute() for path in driver.CONTEXTS)
    assert "torch.backends.cudnn.enabled = False" in source
    assert "require_cuda_recurrent_cell(\"cuda\")" in source
    assert 'candidate_generalization_regime="shared_candidate_in_support"' in source
    assert "observability_suite=True, history_gate=True" in source
