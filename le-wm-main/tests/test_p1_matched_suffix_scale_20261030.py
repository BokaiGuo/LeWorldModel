import numpy as np
import pytest

from experiments.spectral_reachability.p1_matched_suffix_scale_20261030 import make_probes


def test_fixed_order_bitwise_scale_and_bounds():
    a = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
    b = -a
    probes = make_probes(a, b)
    assert len(probes) == 21
    assert [p["index"] for p in probes] == list(range(21))
    assert probes[0]["name"] == "zero"
    assert bytes.fromhex(probes[1]["raw5_float32_hex"]) == (a * np.float32(.25)).tobytes()
    assert bytes.fromhex(probes[9]["raw5_float32_hex"]) == (b * np.float32(.25)).tobytes()
    assert probes[17]["name"] == "+x"
    assert probes == make_probes(a, b)


def test_reject_invalid_actions():
    with pytest.raises(ValueError, match="source actions"):
        make_probes(np.full((5, 2), 1.1), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="source actions"):
        make_probes(np.full((5, 2), np.nan), np.zeros((5, 2)))
