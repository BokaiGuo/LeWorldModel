import json

import pytest

from experiments.spectral_reachability import p1_response_h10_control_binding_20261031 as binding


def sources():
    return [json.loads(path.read_text()) for path in (binding.NEW, binding.OLD, binding.PAIR)]


def test_real_sealed_sources_match_all_control_links():
    result = binding.verify(*sources())
    assert result["control_links"] == 28
    assert all(row["matched_suffixes"] == 42 and row["matched_warmup_frames"] == 33
               for row in result["matches"])
    assert result["P1_admission"] is False


def test_changed_warmup_snapshot_is_rejected():
    new, old, pairing = sources()
    old["anchors"][1]["trace"][0]["receipt"]["state"][0] += 1
    with pytest.raises(Exception):
        binding.verify(new, old, pairing)


def test_changed_control_suffix_is_rejected():
    new, old, pairing = sources()
    old["probe_banks"][0][0]["raw10_float32_hex"] = "00000000"
    with pytest.raises(Exception):
        binding.verify(new, old, pairing)
