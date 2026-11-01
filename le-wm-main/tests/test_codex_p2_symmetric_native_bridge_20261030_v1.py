from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_symmetric_native_bridge_20261030_v1 as b


def test_actual_four_known_roots_bind_without_model():
    _, protocol, cases = b.known_roots()
    assert len(cases) == 4
    assert len({c["context_sha256"] for c in cases}) == 4
    assert b.preflight()["production_ready"] is False
    rows = b.host.frontier.read(protocol["context"]["compilation"])["rows"]
    changed = deepcopy(rows)
    changed[0]["compiled"]["goal_vectors"][0][0] += 1
    with pytest.raises(ValueError):
        b.bind_cases(protocol, changed)


def test_h3_l2_domain_fail_closed():
    assert b.validate_logits(np.zeros(30), 2.).shape == (30,)
    with pytest.raises(ValueError):
        b.validate_logits(np.full(30, 1.), 2.)
    with pytest.raises(ValueError):
        b.validate_logits(np.full(30, np.nan), 2.)
