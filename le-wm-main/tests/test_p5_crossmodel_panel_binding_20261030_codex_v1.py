import copy

import pytest

from experiments.spectral_reachability import p5_crossmodel_panel_binding_20261030_codex_v1 as m


@pytest.fixture(scope="module")
def sources():
    return {k: m.read_pinned(m.ROOT / path, h) for k, (path, h) in m.FILES.items()}


@pytest.fixture
def row(sources):
    return [copy.deepcopy(sources["bridge"]["requests"][0]),
            copy.deepcopy(sources["physical"]["rows"][0]),
            copy.deepcopy(sources["join"]["analysis"]["rows"][0]),
            copy.deepcopy(sources["bridge"]["stats"])]


def test_full_real_fixed_panel(sources):
    result = m.build(**sources)
    assert (len(result["rows"]), len(result["pairs"])) == (164, 80)
    assert result["baseline_indices"] == [0, 41, 82, 123]
    assert result["P5_admission"] is False


@pytest.mark.parametrize("kind", ["identity", "action", "warmup", "image", "time", "label",
                                  "proprio", "normalization", "incomplete", "overlay"])
def test_row_mutations_fail_closed(row, kind):
    q, p, j, stats = row
    if kind == "identity":
        q["task_id"] = "wrong"
    elif kind == "action":
        q["commands"][5][0] += 1
    elif kind == "warmup":
        q["commands"][0][0] += 1
    elif kind == "image":
        q["images"][0]["file_sha256"] = "wrong"
    elif kind == "time":
        q["images"][1]["step"] = 5
    elif kind == "label":
        q["label"]["file_sha256"] = "wrong"
    elif kind == "proprio":
        q["raw_proprio"][0][0] += 1
    elif kind == "normalization":
        q["packets"][0][0] += 1
    elif kind == "incomplete":
        p["own_returned"] = 9
    elif kind == "overlay":
        q["images"][0]["overlay_index"] = 1
    with pytest.raises(ValueError):
        m.bind_row(0, q, p, j, stats, verify_images=False)


def test_png_bytes_checked(row, tmp_path):
    q, p, j, stats = row
    with pytest.raises(ValueError, match="physical image path"):
        m.bind_row(0, q, p, j, stats, panel_dir=tmp_path)


def test_pair_schedule_rejects(sources):
    altered = dict(sources)
    altered["bridge"] = dict(sources["bridge"])
    altered["bridge"]["protocol"] = copy.deepcopy(sources["bridge"]["protocol"])
    altered["bridge"]["protocol"]["pairs"][20]["plus_index"] += 1
    with pytest.raises(ValueError, match="pair schedule"):
        m.build(**altered, verify_images=False)


def test_hash_and_duplicate_json(tmp_path):
    p = tmp_path / "data"
    p.write_bytes(b'{"x":1,"x":2}')
    with pytest.raises(ValueError, match="input hash"):
        m.read_pinned(p, "0" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        m.read_pinned(p, m.sha(p.read_bytes()))


def test_packed_iterator_and_nonfinite():
    assert m.packed(iter([1.0, 2.0]), "<f") == m.packed([1.0, 2.0], "<f")
    with pytest.raises(ValueError):
        m.packed([float("nan")], "<f")
