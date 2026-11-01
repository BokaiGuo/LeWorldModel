import numpy as np
import pytest

from experiments.spectral_reachability import audit_reacher_d2p_qpos_successor_fidelity as audit
from experiments.spectral_reachability.audit_reacher_d2p_qpos_successor_fidelity import _sign_agreement


def test_successor_fidelity_zero_relative_sign_agreement_is_paired():
    assert _sign_agreement(np.asarray([1.0, 2.0, 0.5]), np.asarray([3.0, 4.0, 2.0]))
    assert not _sign_agreement(np.asarray([1.0, 2.0]), np.asarray([3.0, 2.0]))
    with pytest.raises(ValueError, match="paired"):
        _sign_agreement(np.asarray([1.0]), np.asarray([1.0]))


def test_official_source_root_mode_uses_source_native_context(monkeypatch, tmp_path):
    expected = {"context_mode": "official_source_state_history", "warmup_policy": "official_source_state_history_v3_goal_bound_before_history_render"}
    monkeypatch.setattr(audit, "_source_state_context", lambda env, dataset, seed: expected)
    context, index = audit._root_context(
        mode="official_source_state_history", env=object(), dataset=tmp_path / "source.h5",
        seed=3, histories=np.zeros((1, 3, 5, 2), dtype=np.float32),
    )
    assert context is expected
    assert index is None
