import sys
from types import ModuleType

import pytest
import torch

from experiments.pc_loaded_source_binding_20261107_codex import LoadedSourceBinding, SourceBoundCheckpointRGB
from experiments.pc_input_permissions import RGB, freeze_audit


def register(monkeypatch, name, path):
    module = ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)


def test_binding_detects_source_change(tmp_path, monkeypatch):
    source = tmp_path / "model.py"
    source.write_text("value = 1\n")
    register(monkeypatch, "pc_test_source_model", source)
    binding = LoadedSourceBinding([tmp_path])
    assert binding.verify()["modules"]["pc_test_source_model"]["path"] == str(source)
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="changed=.*pc_test_source_model"):
        binding.verify()


def test_binding_detects_new_and_removed_modules(tmp_path, monkeypatch):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("pass\n")
    second.write_text("pass\n")
    register(monkeypatch, "pc_test_source_first", first)
    binding = LoadedSourceBinding([tmp_path])
    register(monkeypatch, "pc_test_source_second", second)
    with pytest.raises(ValueError, match="added=.*pc_test_source_second"):
        binding.verify()
    monkeypatch.delitem(sys.modules, "pc_test_source_second")
    monkeypatch.delitem(sys.modules, "pc_test_source_first")
    with pytest.raises(ValueError, match="removed=.*pc_test_source_first"):
        binding.verify()


def test_missing_loaded_source_fails_closed(tmp_path, monkeypatch):
    register(monkeypatch, "pc_test_missing", tmp_path / "missing.py")
    with pytest.raises(ValueError, match="loaded source missing"):
        LoadedSourceBinding([tmp_path])


def test_synthetic_relative_module_path_is_not_cwd_source(tmp_path, monkeypatch):
    register(monkeypatch, "pc_test_fake_torch_namespace", "_classes.py")
    source = tmp_path / "real.py"
    source.write_text("pass\n")
    register(monkeypatch, "pc_test_real_source", source)
    binding = LoadedSourceBinding([tmp_path])
    assert "pc_test_fake_torch_namespace" not in binding.verify()["modules"]


def test_combined_wrapper_rejects_source_drift(tmp_path, monkeypatch):
    source = tmp_path / "runner.py"
    source.write_text("pass\n")
    register(monkeypatch, "pc_test_bound_runner", source)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.eval()

        def encode(self, inputs):
            return {"emb": inputs["pixels"] * self.weight}

    model = Model()
    weights = tmp_path / "weights.pt"
    config = tmp_path / "config.json"
    torch.save(model.state_dict(), weights)
    config.write_text("{}")
    audit = freeze_audit(RGB, "lewm", [weights, config], preprocessing="fixture", clock="fixture")
    wrapper = SourceBoundCheckpointRGB(model, audit, weights, config, [tmp_path])
    wrapper.encode({"pixels": torch.ones(1)})
    assert wrapper.finish()["checkpoint"]["runtime_receipts"] == 1
    source.write_text("changed = True\n")
    with pytest.raises(ValueError, match="loaded source inventory changed"):
        wrapper.finish()
