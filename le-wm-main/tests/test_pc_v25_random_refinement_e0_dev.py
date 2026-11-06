from experiments import pc_v25_random_refinement_e0_dev_20261104_codex as panel


def test_task_roster_is_unique_and_outside_enumerated_panels():
    tasks = panel.tasks()
    assert len(tasks) == 8
    assert len({(tuple(task["start"]), tuple(task["goal"])) for task in tasks}) == 8
    old = (*panel.prior.prior.prior_panels(), panel.prior.prior.tasks(),
           panel.prior.prospective.tasks(), panel.prior.prospective.cost.tasks(), panel.prior.tasks())
    assert not any(task in previous for task in tasks for previous in old)
