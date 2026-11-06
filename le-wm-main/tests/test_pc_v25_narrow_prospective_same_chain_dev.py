import pytest

from experiments import pc_v25_narrow_prospective_same_chain_dev_20261104_codex as panel


def test_task_panel_is_fixed_and_unique():
    tasks = panel.tasks()
    assert len(tasks) == len({str(task) for task in tasks}) == 16
    assert all(len(task["start"]) == 3 and len(task["goal"]) == 2 for task in tasks)


def test_intervention_fails_closed_without_sealed_plans(tmp_path, monkeypatch):
    old = panel.previous.OUT
    monkeypatch.setattr(panel, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError):
        panel.invoke(panel.previous.intervene)
    assert panel.previous.OUT == old
    assert list(tmp_path.iterdir()) == []
