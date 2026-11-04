import json

import pytest

from experiments import pc_successor_rgb_bound_runner_20261103_codex as binding


def test_receipt_is_exclusive_and_detects_source_drift(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    receipt = tmp_path / "before.json"
    source.write_text("first")
    output.write_text("result")
    before = binding.snapshot([source])
    binding.exclusive_json(receipt, {"sources": before})
    with pytest.raises(FileExistsError):
        binding.exclusive_json(receipt, {})
    monkeypatch.setattr(binding, "PRE", receipt)
    result = binding.finish(before, [source], [output])
    assert result["source_before_sha256"] == binding.digest(receipt)
    assert result["outputs"][str(output.resolve())] == binding.digest(output)
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        binding.finish(before, [source], [output])
    assert json.loads(receipt.read_text())["sources"] == before


def test_missing_duplicate_and_directory_fail_closed(tmp_path):
    source = tmp_path / "source"
    source.write_text("x")
    with pytest.raises(ValueError):
        binding.snapshot([source, source])
    with pytest.raises(ValueError):
        binding.snapshot([tmp_path])
    with pytest.raises(FileNotFoundError):
        binding.snapshot([tmp_path / "missing"])


def test_runtime_helpers_are_in_declared_source_set():
    sources = {path.resolve() for path in binding.source_paths()}
    for name in ("encode_with_state", "estimate_state", "causal_state", "projected_cost",
                 "get_encoder", "load_model_and_dataset", "parameter_hash", "require_cuda"):
        assert binding.Path(getattr(binding.h, name).__code__.co_filename).resolve() in sources
    assert binding.Path(binding.old.__file__).resolve() in sources
    assert binding.Path(binding.scene.__file__).resolve() in sources
