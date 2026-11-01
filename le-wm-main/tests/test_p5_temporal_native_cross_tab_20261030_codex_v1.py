import copy

import pytest

from experiments.spectral_reachability import p5_temporal_native_cross_tab_20261030_codex_v1 as m


@pytest.fixture
def panel():
    t = m.load(m.TEMPORAL, m.PINS[0])
    n = m.load(m.NATIVE, m.PINS[1])
    return t, n


def test_full_fixed_panel(panel):
    r = m.analyze(*panel)
    assert len(r["pairs"]) == 80
    for group, denominator in (("all", 80), ("slot0", 20), ("slot1", 20),
                               ("slot2", 20), ("slot3", 20)):
        for table in r["tables"][group].values():
            assert table["denominator"] == denominator
            assert sum(sum(x.values()) for x in table["counts"].values()) == denominator
    assert not r["P5_admission"]


@pytest.mark.parametrize("kind", ["missing", "permuted", "branch", "cosine", "admission", "native"])
def test_fails_closed(panel, kind):
    t, n = copy.deepcopy(panel)
    if kind == "missing":
        t["pairs"].pop()
    elif kind == "permuted":
        n["pairs"][0], n["pairs"][1] = n["pairs"][1], n["pairs"][0]
    elif kind == "branch":
        t["rows"][1]["task_id"] = "wrong"
    elif kind == "cosine":
        t["pairs"][0]["cells"]["T_AA"]["cosine"] = float("nan")
    elif kind == "native":
        n["pairs"][0]["direction_agreement"]["native_distance"] = "opposite"
    else:
        n["P5_admission"] = True
    with pytest.raises(ValueError):
        m.analyze(t, n)


def test_null_and_zero():
    assert m.sign(None) == "null"
    assert m.sign(-0.0) == "zero"
    with pytest.raises(ValueError):
        m.sign(True)


def test_hash_rejected(tmp_path):
    p = tmp_path / "x"
    p.write_text("{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.load(p, "0" * 64)
