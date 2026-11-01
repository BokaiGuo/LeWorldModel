import copy

import pytest

from experiments.spectral_reachability import p5_rgb_causal_panel_bridge_20261030_codex_v1 as m


@pytest.fixture(scope="module")
def data():
    return {k: m.read_fixed(k) for k in m.FILES}


@pytest.fixture(scope="module")
def source_row(data):
    physical = m.read_json(data["physical_receipt"])["rows"][0]
    panel = m.read_json(data["panel"])["rows"][0]
    return physical, panel


def test_cpu_load_real_checkpoint(data, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    receipt = m.cpu_load(data["checkpoint"])
    assert receipt["strict_load"] and receipt["device"] == "cpu"
    assert receipt["state_dict_entries"] == 26 and receipt["parameter_count"] == 93842
    assert receipt["requires_grad"] is False


def test_cuda_isolation(data, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CPU isolation"):
        m.cpu_load(data["checkpoint"])


def test_actual_input_and_no_label(source_row):
    row = m.bind_row(0, *source_row)
    inputs = m.materialize_inputs(row)
    assert set(inputs) == {"previous_image", "image", "action"}
    assert inputs["previous_image"].shape == inputs["image"].shape == (1, 3, 64, 64)
    assert inputs["action"].shape == (1, 2)
    assert (row["previous_global_step"], row["current_global_step"], row["target_global_step"]) == (14, 15, 16)


@pytest.mark.parametrize("error", ["identity", "action", "current", "missing", "overlay", "incomplete"])
def test_bad_rows_rejected(source_row, error):
    physical, panel = copy.deepcopy(source_row)
    if error == "identity":
        panel["task_id"] = "wrong"
    elif error == "action":
        physical["plan"]["commands"][5][0] += 1
    elif error == "current":
        panel["history_rgb"][2]["file_sha256"] = "0" * 64
    elif error == "missing":
        physical["trajectory"] = [f for f in physical["trajectory"] if not
                                  (f["phase"] == "plan" and f["step"] == 4)]
    elif error == "overlay":
        for f in physical["trajectory"]:
            if f["phase"] == "plan" and f["step"] == 5:
                f["overlay_index"] = 9
    else:
        physical["own_returned"] = 9
    with pytest.raises(ValueError):
        m.bind_row(0, physical, panel)


@pytest.mark.parametrize("error", ["image_hash", "action", "image_path"])
def test_materialize_detects_tamper(source_row, error):
    row = m.bind_row(0, *source_row)
    if error == "image_hash":
        row["current_rgb"]["model_input_chw_float32_sha256"] = "0" * 64
    elif error == "action":
        row["action_raw_float32"][0] += 1
    else:
        row["previous_rgb"]["path"] = "/tmp/invalid.png"
    with pytest.raises(ValueError):
        m.materialize_inputs(row)


def test_target_not_read(source_row):
    physical, panel = copy.deepcopy(source_row)
    for frame in physical["trajectory"]:
        if frame["phase"] == "plan" and frame["step"] == 6:
            frame["png"] = "nonexistent_future.png"
    row = m.bind_row(0, physical, panel)
    assert row["target_identity_only"]["plan_step"] == 6


def test_duplicate_json():
    with pytest.raises(ValueError):
        m.read_json(b'{"x":1,"x":2}')
