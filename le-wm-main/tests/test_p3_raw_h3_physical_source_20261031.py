import json

import pytest

from experiments.spectral_reachability import p3_raw_h3_physical_source_20261031 as source


def test_real_manifest_reconstructs_fixed_first_batch():
    p, data = source.reconstruct(0)
    manifest = json.loads(source.MANIFEST.read_bytes())
    assert p["plans"] == manifest["plans"][:6]
    assert p["manifest_sha256"] == source.MANIFEST_SHA
    assert p["bindings"][str(source.MANIFEST)] == source.MANIFEST_SHA
    assert len(data["references"]) == 2
    assert p["physical_execution"] is False and p["scientific_pass"] is False


@pytest.mark.parametrize("index", [-1, 12, True, "0"])
def test_invalid_batch_index_rejected_before_reading(index):
    with pytest.raises(ValueError, match="batch index"):
        source.reconstruct(index)
