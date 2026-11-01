"""Non-neural protocol/header/archive checks; no model construction."""
import importlib.util
from pathlib import Path
import tarfile

import pytest

PATH=Path(__file__).resolve().parents[1]/"experiments/spectral_reachability/p5_jepa_official_equivalence_20261027_v6.py"
spec=importlib.util.spec_from_file_location("v6",PATH)
v6=importlib.util.module_from_spec(spec); spec.loader.exec_module(v6)


def test_protocol_fixed():
    p=v6.protocol()
    assert p["episodes"]==list(range(1,9)) and p["frames"]==[0,5,10,15]
    assert p["official_config"]["img_size"]==518
    assert p["official_config"]["interpolate_offset"]==.1
    assert not p["official_config"]["interpolate_antialias"]
    assert not p["official_config"]["pretrained"]
    assert p["gates"]==dict(preprocess_max_abs=1e-6,feature_max_abs=1e-4,feature_rmse=1e-5)
    assert p["counts"]==dict(official_forward=8,HF_forward=8)
    assert not p["P5_admission"]


def test_qkv_order_and_coverage():
    m=v6.mapping(); keys=[s for r in m.values() for s in r["sources"]]
    assert len(keys)==len(set(keys))==223
    assert len(m)==175
    for i in range(12):
        r=m[f"blocks.{i}.attn.qkv.weight"]
        assert r["sources"]==[f"encoder.layer.{i}.attention.attention.{q}.weight" for q in ("query","key","value")]
        assert r["shape"]==[1152,384]
    assert m["pos_embed"]["shape"]==[1,1370,384]


@pytest.mark.parametrize("name",["/abs","../escape",f"dinov2-{v6.COMMIT}/../escape","wrongroot/foo"])
def test_traversal(name):
    with pytest.raises(ValueError): v6.safe_member(tarfile.TarInfo(name))


@pytest.mark.parametrize("kind",[tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.FIFOTYPE,tarfile.CHRTYPE])
def test_special(kind):
    m=tarfile.TarInfo(f"dinov2-{v6.COMMIT}/foo");m.type=kind
    with pytest.raises(ValueError): v6.safe_member(m)


def test_size():
    m=tarfile.TarInfo(f"dinov2-{v6.COMMIT}/foo");m.size=8*1024**2+1
    with pytest.raises(ValueError): v6.safe_member(m)


def test_meaningful_key_shape_errors():
    with pytest.raises(ValueError,match="missing.*needed.*unexpected.*extra.*mismatched.*wrong"):
        v6.compare_headers(dict(extra=[1],wrong=[2]),dict(needed=[1],wrong=[3]))


def test_local_header_only():
    report=v6.header_audit(PATH.parents[2])
    assert report["exact_source_coverage"]


def test_no_unsafe_pickle_or_model_prepare():
    import ast
    tree=ast.parse(PATH.read_text())
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="prepare")
    text=ast.get_source_segment(PATH.read_text(),prepare)
    assert "Dinov2Model(" not in text and "dinov2_vits14(" not in text
    assert "torch.load(" not in PATH.read_text() and "pickle" not in PATH.read_text()
