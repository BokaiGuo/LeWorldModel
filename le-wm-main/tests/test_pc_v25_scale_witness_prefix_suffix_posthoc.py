import pytest

from experiments import pc_v25_scale_witness_prefix_suffix_posthoc_20261104_codex as diagnostic


def test_witness_identity_and_success_required():
    task = diagnostic.source.tasks()[0]
    witnesses = {"rows": [{"root": 0, "task": task, "witnesses": [
        {"success": True, "actions": [[0.0, 0.0]] * 15},
        {"success": True, "actions": [[0.1, 0.0]] * 15}]}]}
    assert diagnostic.prefixes_for(0, task, witnesses).shape == (2, 5, 2)
    witnesses["rows"][0]["witnesses"][0]["success"] = False
    with pytest.raises(ValueError, match="verified"):
        diagnostic.prefixes_for(0, task, witnesses)
