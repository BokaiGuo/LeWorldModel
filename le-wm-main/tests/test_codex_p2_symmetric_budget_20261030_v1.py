import numpy as np
from experiments.spectral_reachability import codex_p2_symmetric_budget_20261030_v1 as m


class Backend:
    def rollout(self, context, u):
        terminal = np.pad(u, (0, 162))
        return [context] * 3, terminal

    def jacobian(self, context, u):
        return np.pad(np.eye(30), ((0, 162), (0, 0)))


def test_symmetric_real_work_and_nontrivial_policies():
    results = [m.plan(Backend(), "root", np.ones(192), a) for a in m.ARMS]
    assert all(r["status"] == "complete" for r in results)
    assert all(r["counts_requested"] == m.EXPECTED for r in results)
    assert results[0]["winner"]["cost"] < results[1]["winner"]["cost"]
    assert all(r["equal_total_compute"] is False and r["flops"] is None for r in results)


def test_failure_charged_and_no_export():
    class Broken(Backend):
        def jacobian(self, context, u):
            raise RuntimeError("failed")
    r = m.plan(Broken(), "root", np.ones(192), "joint")
    assert r["status"] == "partial" and r["export"] is None
    assert r["counts_requested"]["jacobian"] == 1
    assert r["events"][-1]["state"] == "attempt"


def test_wall_refusal_no_dispatch():
    ticks = iter([0., 450.])
    r = m.plan(Backend(), "root", np.ones(192), "joint", lambda: next(ticks))
    assert r["status"] == "partial" and r["counts_requested"]["predict"] == 0
    assert r["error"]["type"] == "TimeoutError"


def test_invalid_arm_and_goal():
    import pytest
    with pytest.raises(ValueError):
        m.plan(Backend(), "root", np.ones(192), "other")
    with pytest.raises(ValueError):
        m.plan(Backend(), "root", np.ones(191), "joint")
