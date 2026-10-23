from __future__ import annotations

import inspect

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v17_outcome as audit


def test_v17_outcome_audit_is_read_only_root_clustered_and_cuda_only() -> None:
    assert audit.MARGIN == 0.003
    assert audit.BOOTSTRAP_DRAWS == 10_000
    assert "visible_high_velocity" in str(audit.PANEL)
    source = inspect.getsource(audit.audit)
    assert "no CPU fallback" in source
    assert "generated_after_frozen_selection_sha256" in source
    assert "_root_pair_metrics" in source
    assert "_paired_root_bootstrap" in source
    assert "velocity_matched_braking_pairs" in source


def test_v17_outcome_audit_does_not_derive_uncertainty_from_individual_pairs() -> None:
    source = inspect.getsource(audit.audit)
    assert "recurrent_root" in source
    assert "action_root" in source
    assert 'key="sign_accuracy"' in source
    assert "np.quantile" not in source  # confidence intervals live in the root-cluster helper
