import pytest
from experiments.spectral_reachability import codex_p2_program_independent_prepare_20261030_v1 as p


def test_real_material_has_new_independent_contract():
    m=p.material()
    assert m["protocol"]["per_prefix_per_arm_caps"]["packet"]==36
    assert len(m["cases"])==4 and m["prepare_only"] and not m["scientific_pass"]
    assert any(x["path"].endswith("/weights.pt") for x in m["sources"])


def test_prepare_rejects_nonexclusive_wrong_path(tmp_path):
    with pytest.raises(ValueError):
        p.prepare(tmp_path/"bad")
