import ast
import inspect
import json

import numpy as np
import pytest

from experiments import pc_pointmaze_multimodal_objective_20261107_codex as experiment


def test_scoring_does_not_access_physical_outcome_file():
    tree = ast.parse(inspect.getsource(experiment.score))
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "PRIVATE" not in attrs


def test_audit_uses_sealed_choices_and_preserves_all_roots(tmp_path, monkeypatch):
    public = tmp_path / "public"
    protocol = tmp_path / "protocol"
    config = tmp_path / "config"
    prior_selection = tmp_path / "old_selection"
    private = tmp_path / "physical.npz"
    selected = tmp_path / "selection.json"
    output = tmp_path / "audit.json"
    for file in (public, protocol, config, prior_selection):
        file.write_text(file.name)
    traces = np.zeros((16, 3, 2, 6, 4))
    traces[::2, 0, :, -1, 0] = 2
    traces[::2, 1, :, -1, 0] = 1
    traces[::2, 2, :, -1, 0] = 3
    np.savez(private, traces=traces)
    rows = [{"root": root, "camera": camera, "visual_choice": 0, "combined_choice": 1}
            for root in range(1884, 1892) for camera in range(2)]
    record = {"rows": rows, "public_sha256": experiment.file_hash(public),
              "protocol_sha256": experiment.file_hash(protocol), "config_sha256": experiment.file_hash(config),
              "prior_selection_sha256": experiment.file_hash(prior_selection),
              "runner_sha256": experiment.file_hash(experiment.__file__)}
    selected.write_text(json.dumps(record))
    for name, value in (("PUBLIC", public), ("PROTOCOL", protocol), ("CONFIG", config),
                        ("SELECT", selected), ("AUDIT", output)):
        monkeypatch.setattr(experiment, name, value)
    monkeypatch.setattr(experiment.prior, "SELECT", prior_selection)
    monkeypatch.setattr(experiment.prior, "PRIVATE", private)
    experiment.audit()
    result = json.loads(output.read_text())
    assert len(result["rows"]) == 16
    assert result["improved_roots"] == 8
    assert result["mean_regret"] == {"visual": 1.0, "combined": 0.0}
    assert result["development_exit_met"] is True
    with pytest.raises(FileExistsError):
        experiment.audit()
