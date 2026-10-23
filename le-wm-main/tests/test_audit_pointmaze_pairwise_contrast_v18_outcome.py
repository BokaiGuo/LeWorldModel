from __future__ import annotations

import inspect

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v18_outcome as audit


def test_v18_outcome_audit_is_read_only_root_clustered_and_cuda_only() -> None:
    assert audit.MARGIN == 0.003
    assert audit.BOOTSTRAP_DRAWS == 10_000
    assert "v18_visible_high_velocity_fixed" in str(audit.PANEL)
    source = inspect.getsource(audit.audit)
    assert "no CPU fallback" in source
    assert "generated_after_frozen_selection_sha256" in source
    assert "source_history_model" in source
    assert "_root_pair_metrics" in source
    assert "_paired_root_bootstrap" in source


def test_v18_outcome_audit_uses_root_level_uncertainty() -> None:
    source = inspect.getsource(audit.audit)
    assert "recurrent_root" in source
    assert "action_root" in source
    assert 'key="sign_accuracy"' in source
    assert "np.quantile" not in source
