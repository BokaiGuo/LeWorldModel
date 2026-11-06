import numpy as np
import pytest

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as runner


def test_intervention_hook_restores_endpoint_after_failure(monkeypatch):
    task = runner.tasks()[0]
    row = {"root": 0, "task": task}
    original = runner.mechanism.exact.endpoints
    seen = []

    def recorder(actions):
        seen.append(actions.copy())
        return np.zeros((len(actions), 2)), np.zeros(len(actions))

    def fail(_row):
        runner.mechanism.exact.endpoints(task, np.zeros((1, 5, 2), np.float32))
        raise RuntimeError("injected")

    monkeypatch.setattr(runner.mechanism, "calculate", fail)
    with pytest.raises(RuntimeError, match="injected"):
        runner.calculate_with_journal(row, recorder)
    assert runner.mechanism.exact.endpoints is original
    assert len(seen) == 1


def test_predeclared_panel_distinct_from_recent_panels():
    assert len({str(task) for task in runner.tasks()}) == 8
    assert not any(task in other for task in runner.tasks()
                   for other in (*runner.fixed.prior_panels(), runner.fixed.tasks(), runner.cost.tasks()))
