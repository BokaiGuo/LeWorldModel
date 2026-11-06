import json

import pytest

from experiments.pc_seeded_candidate_counterfactual_20261105_codex import run


def test_counterfactual_refuses_incomplete_physical_parity(tmp_path):
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"n_tasks": 50, "n_parity": 49, "n_errors": 0}))
    output = tmp_path / "out.json"
    with pytest.raises(ValueError, match="parity"):
        run(tmp_path / "missing.h5", tmp_path / "missing.jsonl", parity, output)
    assert not output.exists()
