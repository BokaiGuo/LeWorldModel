from copy import deepcopy

import pytest

from experiments.spectral_reachability import p2_knownroot_plan_overlap_20261102_exclusive as m


def examples():
    join = dict(status="joined_not_physical_ready", physical_factory_bound=False,
                physical_authorized=False, P1_admission=False, P2_admission=False, plans=[])
    physical = {"rows": []}
    for index, root in enumerate(m.ROOTS):
        for arm in m.ARMS:
            commands = [[index, k] for k in range(10)]
            join["plans"].append(dict(root_index=index, root_sha256=root, arm=arm,
                                      commands=deepcopy(commands), history_rgb_sha256="h",
                                      goal_rgb_sha256="g"))
            physical["rows"].append(dict(
                root_index=index, root_sha256=root, method=arm, status="completed",
                full_horizon_completed=True, failure=None, planned_commands=10,
                completed_commands=10, physical_env_step_attempts=10, physical_env_step_returns=10,
                context_identity=dict(history_rgb_sha256="h", goal_rgb_sha256="g"),
                trace=[dict(primitive_step=k + 1, kind="observation", command=c)
                       for k, c in enumerate(commands)]))
    return join, physical


def test_complete_overlap_and_difference():
    join, physical = examples()
    assert m.audit(join, physical)["identical_commands"] == 8
    join["plans"][1]["commands"][0] = [8, 8]
    report = m.audit(join, physical)
    assert (report["identical_commands"], report["different_commands"]) == (7, 1)
    assert len(report["fixed_eight_branch_census"]) == 8


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "partial", "rgb", "authorized"])
def test_reject_invalid_inputs(mutation):
    join, physical = examples()
    if mutation == "missing":
        join["plans"].pop()
    elif mutation == "duplicate":
        join["plans"][-1] = deepcopy(join["plans"][0])
    elif mutation == "partial":
        physical["rows"][0]["physical_env_step_returns"] = 9
    elif mutation == "rgb":
        join["plans"][0]["goal_rgb_sha256"] = "other"
    else:
        join["physical_authorized"] = True
    with pytest.raises(ValueError):
        m.audit(join, physical)
