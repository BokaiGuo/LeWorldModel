from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import e3_cp10_first_packet_saved_20261031 as saved


def test_fixed_saved_vectors_and_tampering():
    data = json.loads(saved.RESULT.read_bytes())
    rows = saved.verify(data)
    assert len(rows) == 8
    corrupt = deepcopy(data)
    corrupt["rows"][0]["prediction"][0] += 10
    with pytest.raises(ValueError, match="score"):
        saved.verify(corrupt)
    corrupt = deepcopy(data)
    corrupt["rows"][0]["arm"] = "random"
    with pytest.raises(ValueError, match="roster"):
        saved.verify(corrupt)
