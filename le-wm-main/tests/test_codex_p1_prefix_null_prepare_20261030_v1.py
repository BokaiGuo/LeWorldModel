import copy

import pytest

from experiments.spectral_reachability import codex_p1_prefix_null_prepare_20261030_v1 as m


@pytest.fixture(scope="module")
def source():
    return m.load()


def test_real_fixed_pairs(source):
    out = m.build(source)
    assert out["logical_plans"] == 8 and len(out["plans"]) == 16
    assert out["physical_authorized"] is False and out["P1_admission"] is False
    assert out["deployment_status"].startswith("blocked")
    for i in range(8):
        selected, zero = out["plans"][2*i:2*i+2]
        a = bytes.fromhex(selected["raw15_float32_hex"])
        b = bytes.fromhex(zero["raw15_float32_hex"])
        assert selected["role"] == "selected_prefix" and zero["role"] == "zero_prefix"
        assert len(a) == len(b) == 120 and b[:40] == bytes(40) and a[40:] == b[40:]
        assert selected["task_id"] == zero["task_id"]
        assert selected["suffix_sha256"] == zero["suffix_sha256"]
        assert selected["logical_plan"] == zero["logical_plan"] == i
    assert all(r["duplicate_action_task_group"] >= 0 for r in out["plans"])


def test_duplicate_groups_require_same_task_and_actions(source):
    out = m.build(source)
    for a in out["plans"]:
        for b in out["plans"]:
            if a["duplicate_action_task_group"] == b["duplicate_action_task_group"]:
                assert (a["task_id"], a["physical_root_sha256"], a["condition"], a["raw15_float32_hex"]) == (
                    b["task_id"], b["physical_root_sha256"], b["condition"], b["raw15_float32_hex"])


@pytest.mark.parametrize("mutation", ["missing", "role", "arm", "task", "nonfinite", "bounds"])
def test_invalid_source_fails(source, mutation):
    data = copy.deepcopy(source)
    if mutation == "missing":
        data["rows"].pop()
    elif mutation == "role":
        data["rows"][1]["role"] = "incumbent"
    elif mutation == "arm":
        data["rows"][1]["arm"] = "unknown"
    elif mutation == "task":
        data["rows"][1]["task_id"] = "wrong"
    elif mutation == "nonfinite":
        data["rows"][1]["raw15_float32_hex"] = "0000c07f" + data["rows"][1]["raw15_float32_hex"][8:]
    else:
        data["rows"][1]["raw15_float32_hex"] = "00000040" + data["rows"][1]["raw15_float32_hex"][8:]
    with pytest.raises((ValueError, KeyError)):
        m.build(data)


def test_source_pin_rejected(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_text("{}")
    monkeypatch.setattr(m, "JOIN", path)
    with pytest.raises(ValueError):
        m.load()


def test_exclusive_prepare(tmp_path, monkeypatch, source):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "load", lambda: source)
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / (m.NAME + "_prepare_test.json")
    receipt = m.prepare(path)
    assert receipt["physical_authorized"] is False
    assert m.verify(path, receipt["sha256"])["fixed_rows"] == 16
    with pytest.raises(ValueError):
        m.verify(path, "0" * 64)
    changed = copy.deepcopy(source)
    changed["rows"][1]["raw15_float32_hex"] = "0000003f" + changed["rows"][1]["raw15_float32_hex"][8:]
    monkeypatch.setattr(m, "load", lambda: changed)
    with pytest.raises(ValueError, match="drift"):
        m.verify(path, receipt["sha256"])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        m.prepare(path)
    assert path.read_bytes() == original
