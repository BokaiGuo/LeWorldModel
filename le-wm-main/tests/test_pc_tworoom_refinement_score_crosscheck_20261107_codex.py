import json

import numpy as np
import pytest

from experiments import pc_tworoom_refinement_score_crosscheck_20261107_codex as cross


def test_comparison_reports_tie_and_selection_change():
    actions = np.zeros((9, 5, 2), dtype=np.float32)
    actions[1] = 0.1
    old = np.arange(9, dtype=float)
    old[1] = 0.001
    other = old.copy()
    other[0], other[1] = 0.002, 0.0
    result = cross.compare(old, other, actions)
    assert result["within_tolerance"] and result["index_changed"] and result["action_changed"]
    assert result["old_margin"] == pytest.approx(0.001)
    actions[1] = actions[0]
    assert not cross.compare(old, other, actions)["action_changed"]


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        cross.compare(np.full(9, np.nan), np.zeros(9), np.zeros((9, 5, 2)))
    with pytest.raises(ValueError):
        cross.compare(np.zeros(8), np.zeros(9), np.zeros((9, 5, 2)))


def test_existing_result_integrity_and_summary():
    if not cross.OUT.exists():
        pytest.skip("development audit not run")
    result = json.loads(cross.OUT.read_text())
    assert result["plan_sha256"] == cross.file_hash(cross.src.PLAN)
    assert result["pool_sha256"] == cross.file_hash(cross.src.POOL)
    assert [r["root"] for r in result["rows"]] == list(range(10, 50))
    with np.load(cross.src.POOL, allow_pickle=False) as pool:
        for arm in ("adam", "random"):
            results = [cross.compare(pool[f"r{r['root']}_{arm}_scores"], r["arms"][arm]["rescored"],
                                     pool[f"r{r['root']}_{arm}_actions"]) for r in result["rows"]]
            assert result["summary"][arm]["max_abs_error"] == max(r["max_abs_error"] for r in results)
            assert result["summary"][arm]["action_change_roots"] == [
                r["root"] for r, c in zip(result["rows"], results) if c["action_changed"]]
