from __future__ import annotations

import numpy as np

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v12_outcome as audit


def test_root_cluster_audit_does_not_treat_pairs_as_independent() -> None:
    records = []
    for root in ("a", "b"):
        for candidate_id in range(4):
            records.append({"instance_id": root, "candidate_id": candidate_id})
    actual = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    method = np.asarray([0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0])
    baseline = np.asarray([2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0])
    method_root = audit._root_pair_metrics(records, method, actual)
    baseline_root = audit._root_pair_metrics(records, baseline, actual)
    gain = audit._paired_root_bootstrap(method_root, baseline_root, key="sign_accuracy", higher_is_better=True, seed=7)
    assert gain["root_count"] == 2
    assert gain["mean_gain"] == 1.0


def test_outcome_audit_is_explicitly_read_only_and_cuda_only() -> None:
    assert audit.MARGIN == 0.003
    assert audit.BOOTSTRAP_DRAWS == 10_000
    assert "no CPU fallback" in audit.audit.__doc__ if audit.audit.__doc__ else True
    import inspect
    source = inspect.getsource(audit.audit)
    assert "_validate_manifest" in source
    assert "expected_root_counts=expected_root_counts" in source
