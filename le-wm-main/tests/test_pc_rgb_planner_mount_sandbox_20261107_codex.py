import copy

import pytest

from experiments.pc_rgb_planner_mount_sandbox_20261107_codex import compare, make_command


def test_compare_rejects_selected_or_score_drift():
    ref = {"public_sha256": "a", "model_digest": "b",
           "rows": [{"episode": 1, "candidates": [0], "direction": [0], "gradient": [0],
                     "scores": [1], "selected_index": 0}]}
    assert len(compare(copy.deepcopy(ref), ref)) == 1
    bad = copy.deepcopy(ref)
    bad["rows"][0]["scores"] = [2]
    with pytest.raises(ValueError, match="scores"):
        compare(bad, ref)


def test_mounts_do_not_include_dataset_or_all_results(tmp_path):
    command, denied, mounts = make_command(tmp_path)
    assert "--unshare-pid" in command
    assert all("/datasets/" not in src for _, src, _ in mounts)
    assert all(not src.endswith("/results") for _, src, _ in mounts)
    assert any("/datasets/" in path for path in denied)
