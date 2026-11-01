"""Non-neural read/contract tests; never run --validate-cuda here."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/spectral_reachability/p5_jepa_real_forward_20261027_v4.py"
SPEC = importlib.util.spec_from_file_location("p5_real_v4", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def stats():
    return m.source_stats(ROOT / "external/jepa-wms/app/plan_common/datasets/pusht_dset.py")


def arrays():
    # Explicit array contract fixtures, not neural hidden states or efficacy evidence.
    return (np.arange(21*20*5, dtype=np.float32).reshape(21,20,5),
            np.arange(21*20*2, dtype=np.float32).reshape(21,20,2),
            np.arange(21*20*2, dtype=np.float32).reshape(21,20,2))


def test_exact_history_alignment_and_units():
    states, velocities, actions = arrays()
    h = m.histories(states, velocities, actions, stats())
    raw = actions[0, :15]
    expected = (raw / 100 - np.array(stats()["ACTION_MEAN"], np.float32)) / np.array(stats()["ACTION_STD"], np.float32)
    np.testing.assert_array_equal(h["actions_raw"], raw)
    np.testing.assert_allclose(h["action_packets"], expected.reshape(3,10))
    np.testing.assert_array_equal(h["proprio_raw"][:, :2], states[0, [0,5,10,15], :2])
    np.testing.assert_array_equal(h["proprio_raw"][:, 2:], velocities[0, [0,5,10,15]])
    assert h["proprio_normalized"].shape == (4,4)


@pytest.mark.parametrize("part", [0,1,2])
def test_wrong_array_shape_rejected(part):
    data = list(arrays())
    data[part] = data[part][:, :10]
    with pytest.raises(ValueError):
        m.histories(*data, stats())


def test_restricted_array_explicit_flags(tmp_path, monkeypatch):
    p = tmp_path / "array.pt"
    torch.save(torch.arange(5), p)
    original = torch.load
    calls = []
    def spy(*args, **kw):
        calls.append(kw)
        return original(*args, **kw)
    monkeypatch.setattr(torch, "load", spy)
    np.testing.assert_array_equal(m.restricted_array(p, m.sha(p)), np.arange(5))
    assert calls == [{"weights_only": True, "map_location": "cpu"}]


def test_bad_array_hash_rejected_before_loading(tmp_path, monkeypatch):
    p = tmp_path / "bad.pt"
    p.write_bytes(b"not trusted")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="SHA"):
        m.restricted_array(p, "0"*64)


def test_nonfinite_data_rejected(tmp_path):
    p = tmp_path / "nan.pt"
    torch.save(torch.tensor([float("nan")]), p)
    with pytest.raises(ValueError, match="nonfinite"):
        m.restricted_array(p, m.sha(p))


def test_no_dataset_code_executed_for_stats(tmp_path):
    p = tmp_path / "constants.py"
    p.write_text("raise RuntimeError('must not execute')\n" + "\n".join(
        f"{k}=torch.tensor({v!r})" for k,v in stats().items()))
    assert m.source_stats(p) == stats()


def test_cuda_missing_is_hard_failure(monkeypatch):
    monkeypatch.setattr(m, "verify_seal", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA required"):
        m.validate_cuda(ROOT, {})


def test_array_digest_binds_shape_dtype_values():
    a = np.arange(6,dtype=np.float32)
    assert m.array_sha(a) != m.array_sha(a.reshape(2,3))
    assert m.array_sha(a) != m.array_sha(a.astype(np.float64))
    assert m.array_sha(a) != m.array_sha(a+1)


def test_no_pickle_dataset_random_hidden_or_unsafe_allowlists():
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name != "pickle" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"pickle", "app.plan_common.datasets.pusht_dset"}
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"add_safe_globals", "safe_globals", "randn", "rand", "backward"}
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "load":
                    kw = {k.arg:k.value for k in node.keywords}
                    assert kw["weights_only"].value is True


def test_actual_local_v3_and_safe_read_only_real_inputs():
    m.check_v3(ROOT)
    h, rgb, source_stats, n = m.real_inputs(ROOT)
    assert n >= 16 and rgb.shape[0] == 4 and rgb.dtype == np.uint8
    assert np.isfinite(h["action_packets"]).all()
    assert source_stats["ACTION_STD"] == [0.2019,0.2002]
    assert h["action_packets"].shape == (3,10)


def test_source_seal_rejects_drift(monkeypatch):
    monkeypatch.setattr(m, "bindings", lambda root: [{"path":"x", "sha256":"new"}])
    with pytest.raises(ValueError, match="seal drift"):
        m.verify_seal(ROOT, {"schema":m.NAME, "status":"prepared_non_neural", "bindings":[]})


def test_real_rgb_native_preprocessing_matches_reference_without_neural_forward(monkeypatch):
    monkeypatch.syspath_prepend(str(m.native_root(ROOT)))
    from app.plan_common.datasets.transforms import make_transforms
    rgb, _ = m.decode_video(m.data_root(ROOT) / "obses/episode_000.mp4")
    pixels = torch.from_numpy(rgb.copy()).permute(0,3,1,2).float() / 255
    transform = make_transforms(img_size=224, random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.,1.), random_resize_scale=(1.,1.),
        reprob=0., auto_augment=False, motion_shift=False)
    actual = transform(pixels.clone())
    expected = torch.nn.functional.interpolate(pixels, size=(224,224), mode="bilinear", align_corners=False)
    expected = (expected-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
    assert actual.shape == (4,3,224,224)
    assert (actual-expected).abs().max().item() <= 1e-6
