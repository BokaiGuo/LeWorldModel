from copy import deepcopy

import pytest

from experiments.spectral_reachability.consume_p3_fixed_execution_MAIN_20261028_v1 import independent


def fixture():
    tasks = [dict(goal=[0.], projection=[[1.]]) for _ in range(2)]
    branches, rows, winners, prefixes = [], [], [], []
    for slot in (0, 1):
        for mode, horizon in (('fixed', 1), ('adaptive', 3)):
            branches.append(dict(slot=slot, mode=mode, candidates=2))
            for index in (0, 1):
                rows.append(dict(slot=slot, mode=mode, index=index, horizon=horizon,
                    cost=1., actions=[[0., 0.]]*(5*horizon),
                    whole=[dict(latent_history=[[1.]]) for _ in range(horizon)]))
            winners.append(dict(slot=slot, mode=mode, index=0, cost=1.))
            prefixes.append(dict(slot=slot, mode=mode, request=dict(commands=[[0., 0.]]*5)))
    return dict(tasks=tasks, branches=branches, jobs=[{}]*8), dict(rows=rows, winners=winners, prefixes=prefixes)


def test_all_candidates_stable_tie_and_common_execution_boundary():
    panel, report = fixture()
    rows = independent(panel, report)
    assert len(rows) == 4
    assert all(r['selected_index'] == 0 and r['execution_boundary_Q'] == 1. and r['requested_execution_steps'] == 5 for r in rows)


@pytest.mark.parametrize('fault', ['cost', 'winner', 'missing', 'commands'])
def test_saved_mismatch_rejected(fault):
    panel, report = deepcopy(fixture())
    if fault == 'cost':
        report['rows'][0]['cost'] = 0.
    elif fault == 'winner':
        report['winners'][0]['index'] = 1
    elif fault == 'missing':
        report['rows'].pop()
    else:
        report['prefixes'][0]['request']['commands'][0] = [0.1, 0.]
    with pytest.raises(ValueError):
        independent(panel, report)
