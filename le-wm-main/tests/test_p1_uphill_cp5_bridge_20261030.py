import json

from experiments.spectral_reachability import p1_uphill_cp5_bundle_20261030 as bundle
from experiments.spectral_reachability import p1_uphill_cp5_physical20_20261030 as physical


def test_real_bundle_binds_each_prefix_and_checkpoint():
    assert physical.sha(bundle.PREP) == physical.PREP_SHA
    assert physical.sha(bundle.OUT) == physical.BUNDLE_SHA
    prepared = json.loads(bundle.PREP.read_bytes())
    rows = json.loads(bundle.OUT.read_bytes())["rows"]
    assert len(prepared["anchors"]) == len(rows) == 24
    for index, (anchor, row) in enumerate(zip(prepared["anchors"], rows, strict=True)):
        bundle.blind.validate_row(row)
        assert row["index"] == index == anchor["index"]
        assert row["role"] == anchor["role"]
        assert row["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"]
        assert row["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"]
