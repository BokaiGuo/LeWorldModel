import ast
import inspect
import json

import pytest

from experiments import pc_rgb_true_history_isolated_20261107_codex as isolated


def test_scoring_does_not_reference_physical_trace_or_truth():
    tree = ast.parse(inspect.getsource(isolated.score))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not {"TRACE", "TRUTH", "PRED", "RECORD"} & attributes


def test_prepare_exports_only_public_fields(tmp_path, monkeypatch):
    trace = tmp_path / "trace.json"
    record = tmp_path / "record.json"
    inputs = tmp_path / "input"
    protocol = tmp_path / "protocol"
    output = tmp_path / "manifest.json"
    inputs.write_bytes(b"input")
    protocol.write_bytes(b"protocol")
    trace.write_text(json.dumps({
        "input_sha256": isolated.file_hash(inputs),
        "protocol_sha256": isolated.file_hash(protocol),
        "rows": [{"private_state": [1, 2, 3]}],
    }))
    record.write_text(json.dumps({"model_sha256": "model", "action_array_sha256": "actions"}))
    monkeypatch.setattr(isolated.base, "TRACE", trace)
    monkeypatch.setattr(isolated.base, "RECORD", record)
    monkeypatch.setattr(isolated.base, "INPUT", inputs)
    monkeypatch.setattr(isolated.base, "PROTOCOL", protocol)
    monkeypatch.setattr(isolated, "MANIFEST", output)
    isolated.prepare()
    manifest = json.loads(output.read_text())
    assert set(manifest) == {"input_sha256", "protocol_sha256", "model_sha256",
                             "action_array_sha256", "roots", "sides", "scope"}
    assert "private_state" not in output.read_text()
    with pytest.raises(FileExistsError):
        isolated.prepare()
