import json

import numpy as np

from experiments import pc_conditional_selected_alignment_20261107_codex as audit
from experiments import pc_rgb_conditional_temporal_cem_20261107_codex as source
from experiments.pc_input_permissions import file_hash


def test_alignment_artifact_bindings_and_clocks():
    result = json.loads(audit.RESULT.read_text())
    physical = json.loads(audit.PHYS.read_text())
    assert result["features_sha256"] == file_hash(audit.FEATURES)
    assert result["pred_sha256"] == file_hash(audit.PRED)
    assert result["physical_sha256"] == file_hash(audit.PHYS)
    assert physical["source_physical_sha256"] == file_hash(source.PHYS)
    assert [row["root"] for row in result["rows"]] == list(range(20, 50))
    with np.load(audit.FEATURES, allow_pickle=False) as data:
        for row in result["rows"]:
            root = row["root"]
            pred, actual, goal = (data[f"r{root}_{name}"] for name in ("predicted", "actual", "goal"))
            assert pred.shape == actual.shape and pred.shape[:2] == (2, 6)
            np.testing.assert_allclose(np.square(pred[:, 1:] - goal).sum(-1),
                                       row["predicted_cost"], rtol=0, atol=1e-5)
            np.testing.assert_allclose(np.square(actual[:, 1:] - goal).sum(-1),
                                       row["actual_cost"], rtol=0, atol=1e-5)


def test_saved_physical_replay_matches_original_step_by_step():
    new = json.loads(audit.PHYS.read_text())["rows"]
    old = json.loads(source.PHYS.read_text())["rows"]
    for before, after in zip(old, new, strict=True):
        assert before["root"] == after["root"]
        for method in source.METHODS:
            for expected, observed in zip(before["outcomes"][method]["trace"],
                                          after["outcomes"][method], strict=True):
                assert abs(expected["distance"] - observed["distance"]) <= 1e-5
                assert expected["terminated"] == observed["terminated"]
                assert expected["truncated"] == observed["truncated"]
