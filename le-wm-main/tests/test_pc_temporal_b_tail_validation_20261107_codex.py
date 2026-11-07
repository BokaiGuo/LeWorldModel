import numpy as np
import pytest

from experiments.pc_temporal_b_tail_validation_20261107_codex import validate_bank


def test_bank_requires_shared_prefixes_and_common_tail_identity():
    a = np.zeros((8, 8, 25, 2), dtype=np.float32)
    b = a.copy()
    b[:, :, 10:] = 0.5
    validate_bank(a, b)
    broken = b.copy()
    broken[1, 0, 11, 0] = 0.2
    with pytest.raises(ValueError, match="identities"):
        validate_bank(a, broken)


def test_bank_rejects_prefix_mismatch_and_illegal_actions():
    a = np.zeros((8, 8, 25, 2), dtype=np.float32)
    b = a.copy()
    b[:, :, 10:] = 0.5
    b[0, 0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="prefixes"):
        validate_bank(a, b)
    b[0, 0, 0, 0] = 0
    b[0, 0, 10, 0] = 2
    with pytest.raises(ValueError, match="invalid"):
        validate_bank(a, b)
