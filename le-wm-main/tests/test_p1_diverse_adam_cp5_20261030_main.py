import hashlib
import json

import pytest

from experiments.spectral_reachability import p1_diverse_adam_cp5_20261030_main as planner


def test_fixed_new_prefix_inputs_and_original_starts():
    path = planner.ROOT / "results/p1_diverse_adam_cp5_20261030_main_prepare_RUN01.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "cee64cf13f17197b81e1a95a659f76d1cbf880b58638dbdb5adff725e37100c9"
    bundle = json.loads(path.read_bytes())
    h10 = json.loads(planner.H10_PREP.read_bytes())
    assert len(bundle["rows"]) == len(h10["anchors"]) == 3
    for row, anchor in zip(bundle["rows"], h10["anchors"], strict=True):
        planner.blind.validate_row(row)
        assert row["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"]
        assert row["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"]
        assert planner.prior.initial_actions(row["logical_pair"]).shape == (4, 2, 5, 2)


def test_blind_rejects_changed_prefix_shape():
    path = planner.ROOT / "results/p1_diverse_adam_cp5_20261030_main_prepare_RUN01.json"
    row = json.loads(path.read_bytes())["rows"][0]
    row["prefix_raw_f32_hex"] = "00"
    with pytest.raises(ValueError):
        planner.blind.validate_row(row)
