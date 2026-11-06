from experiments import pc_v25_candidate_scale_e0_dev_20261104_codex as experiment


def test_paired_task_roster_and_scale_contract():
    tasks = experiment.tasks()
    assert len(tasks) == 8
    assert len({(tuple(task["start"]), tuple(task["goal"])) for task in tasks}) == 8
    assert experiment.ARMS == {"original": (0.25, 0.5), "smaller": (0.125, 0.25)}
    existing = (*experiment.prior.prior_panels(), experiment.prior.tasks(),
                experiment.prospective.tasks(), experiment.prospective.cost.tasks())
    assert not any(task in panel for task in tasks for panel in existing)
