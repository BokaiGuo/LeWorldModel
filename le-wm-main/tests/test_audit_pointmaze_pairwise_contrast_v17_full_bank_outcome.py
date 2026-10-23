from __future__ import annotations

import inspect

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v17_full_bank_outcome as audit


def test_v17_full_bank_audit_requires_bound_sealed_artifacts_and_cuda() -> None:
    assert audit.BOOTSTRAP_DRAWS == 10_000
    source = inspect.getsource(audit.audit)
    assert "no CPU fallback" in source
    assert "outcome_labels_sha256" in source
    assert "score_report" in source
    assert "velocity_matched_braking_pairs" in source


def test_v17_full_bank_audit_uses_root_level_regret_and_top1_metrics() -> None:
    source = inspect.getsource(audit.audit)
    assert "_root_metrics" in source
    assert 'key="top1_accuracy"' in source
    assert 'key="regret"' in source
    assert "_paired_root_bootstrap" in source
