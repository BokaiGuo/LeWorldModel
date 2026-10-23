from __future__ import annotations

import inspect

from experiments.gdd_wm import audit_pointmaze_pairwise_contrast_v13_position_alias_preflight as v13


def test_v13_declares_fresh_roots_and_position_alias_gate_before_any_fit() -> None:
    assert v13.ROOT_RANGE == (21_384, 22_216)
    assert v13.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    assert v13.MIN_FOLD_BALANCE == 0.40
    assert v13.MAX_PROXY_ACCURACY == 0.625
    source = inspect.getsource(v13.preflight)
    assert "pairwise_contrast_position_aliased_interaction_candidate_banks" in source
    assert "position_only_proxy_rejected" in source
