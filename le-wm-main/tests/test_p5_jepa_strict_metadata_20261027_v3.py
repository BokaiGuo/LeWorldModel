"""Metadata/read tests; no CUDA or model forward."""
import importlib.util
from pathlib import Path
import pickle

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/p5_jepa_strict_metadata_20261027_v3.py"
SPEC = importlib.util.spec_from_file_location("strict_p5_v3", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_load_options_and_meta_storage(tmp_path, monkeypatch):
    path = tmp_path / "weights.pt"
    torch.save({"predictor": {"module.x": torch.empty(2, 3)}}, path)
    original = torch.load
    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", spy)
    value = m.load_meta(path, m.sha(path))
    assert calls == [{"weights_only": True, "map_location": "meta", "mmap": True}]
    tensor = value["predictor"]["module.x"]
    assert tensor.device.type == "meta"
    assert tensor.untyped_storage().data_ptr() == 0
    assert m.nested_metadata(value)[-1]["shape"] == [2, 3]


def test_sha_rejected_before_load(tmp_path, monkeypatch):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"untrusted")
    def reject(*a, **kw):
        pytest.fail("must not load hash mismatch")
    monkeypatch.setattr(torch, "load", reject)
    with pytest.raises(ValueError, match="SHA"):
        m.load_meta(path, "0" * 64)


def test_unsafe_global_rejection_preserved(tmp_path):
    import numpy as np
    path = tmp_path / "numpy.pt"
    torch.save({"stats": np.array([1])}, path)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        m.load_meta(path, m.sha(path))


def test_exact_mismatch_lists():
    actual = {"module.w": torch.empty(16, 2, 1, device="meta"),
              "module.extra": torch.empty(1, device="meta")}
    result = m.compare(actual, {"w": [16, 4, 1], "missing": [1]})
    assert result["missing_keys"] == ["missing"]
    assert result["unexpected_keys"] == ["extra"]
    assert result["shape_mismatches"] == [{"key": "w", "actual": [16, 2, 1], "expected": [16, 4, 1]}]
    assert not result["exact_match"]


def test_dtype_rejected():
    result = m.compare({"x": torch.empty(1, dtype=torch.float64, device="meta")}, {"x": [1]})
    assert result["dtype_mismatches"] and not result["exact_match"]


def test_prefix_collision():
    with pytest.raises(ValueError, match="collision"):
        m.normalize({"module.x": 1, "x": 2})


def test_nonmeta_rejected():
    with pytest.raises(ValueError, match="non-meta"):
        m.nested_metadata(torch.empty(1))


@pytest.mark.parametrize("task", ["pusht", "pointmaze", "wall"])
def test_source_schema_counts(task):
    import yaml
    root = SOURCE.parents[2]
    cfg = yaml.safe_load((root / "external/jepa-wms/configs/vjepa_wm" / m.CONFIGS[task]).read_text())
    shapes = m.expected_schema(cfg, m.PROPRIO_DIMS[task])
    assert len(shapes["predictor"]) == 92
    assert shapes["predictor"]["action_encoder.weight"] == [400, 10]
    assert shapes["proprio_encoder"]["patch_embed.weight"] == [16, m.PROPRIO_DIMS[task], 1]


def test_no_unsafe_load_or_forward_calls():
    import ast
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"add_safe_globals", "safe_globals", "forward"}
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "load":
                kwargs = {k.arg: k.value for k in node.keywords}
                assert isinstance(kwargs["weights_only"], ast.Constant) and kwargs["weights_only"].value is True
