import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_h3_rechart_cuda_20261030_v1 as m


def test_checks_recomputed_tokens_and_domain():
    raw = np.zeros((3,5,2), np.float32)
    norm = np.ones((3,10), np.float32)
    row = dict(root=0, arm="joint", winner=dict(logits=[0.]*30,
               contexts=[dict(action_history=[v.tolist()]) for v in norm]))
    out = m.check_packets(row, raw, norm)
    assert out["original_raw_bits_proved"] is False and len(out["commands"]) == 15
    changed = norm.copy(); changed[1,0] += 1
    with pytest.raises(ValueError): m.check_packets(row, raw, changed)
    raw[0,0,0] = 2
    with pytest.raises(ValueError): m.check_packets(row, raw, norm)
