import json

import pytest

from experiments.pc_prefix_contact_audit_20261105_codex import run


def test_rejects_incomplete_intervention_before_creating_output(tmp_path):
    previous = tmp_path / "previous.json"
    chains = tmp_path / "chains.jsonl"
    output = tmp_path / "out.json"
    previous.write_text(json.dumps({"n_tasks": 50, "n_errors": 1}))
    chains.write_text(json.dumps({"actions": [0] * 50}) + "\n")
    with pytest.raises(ValueError, match="missing complete"):
        run(tmp_path / "missing.h5", chains, previous, output)
    assert not output.exists()
