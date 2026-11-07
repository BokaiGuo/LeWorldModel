import ast
import inspect
import json

import numpy as np
import pytest

from experiments import pc_pointmaze_latent_effect_anchor_20261107_codex as experiment


def test_score_cannot_directly_read_physical_outcomes():
    attrs = {node.attr for node in ast.walk(ast.parse(inspect.getsource(experiment.score)))
             if isinstance(node, ast.Attribute)}
    assert "PRIVATE" not in attrs


def test_audit_keeps_full_panel_and_checks_sealed_choices(tmp_path, monkeypatch):
    inputs, previous, protocol = (tmp_path / name for name in ("input", "previous", "protocol"))
    selected, private, output = (tmp_path / name for name in ("selected", "private.npz", "audit"))
    for file in (inputs, previous, protocol):
        file.write_text(file.name)
    traces = np.zeros((16, 3, 2, 6, 4))
    traces[::2, 0, :, -1, 0] = 2
    traces[::2, 1, :, -1, 0] = 1
    traces[::2, 2, :, -1, 0] = 3
    np.savez(private, traces=traces)
    rows = [{"root": root, "camera": camera, "original_choice": 0, "anchored_choice": 1}
            for root in range(1884, 1892) for camera in range(2)]
    selected.write_text(json.dumps({"rows": rows, "input_sha256": experiment.file_hash(inputs),
                                     "prior_sha256": experiment.file_hash(previous),
                                     "protocol_sha256": experiment.file_hash(protocol),
                                     "runner_sha256": experiment.file_hash(experiment.__file__)}))
    monkeypatch.setattr(experiment.prior, "PUBLIC", inputs)
    monkeypatch.setattr(experiment.prior, "SELECT", previous)
    monkeypatch.setattr(experiment.prior, "PRIVATE", private)
    monkeypatch.setattr(experiment, "PROTOCOL", protocol)
    monkeypatch.setattr(experiment, "SELECT", selected)
    monkeypatch.setattr(experiment, "AUDIT", output)
    experiment.audit()
    result = json.loads(output.read_text())
    assert len(result["rows"]) == 16
    assert result["improved_roots"] == 8
    assert result["mean_regret"] == {"original": 1.0, "anchored": 0.0}
    assert result["development_exit_met"]
    with pytest.raises(FileExistsError):
        experiment.audit()
