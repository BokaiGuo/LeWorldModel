from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import main_nonlinear_suffix_midpoint_actual_analysis_20261028_v1 as m


def fixture():
    root = np.stack([np.full(192, x, np.float32) for x in (0, 1, 2)])
    mid = np.concatenate((root[1:], np.full((1, 192), 3, np.float32)))
    actual = np.full(192, 1, np.float32)
    pred = np.stack([mid[0], mid[1], actual])[None]
    spec = dict(index=0, root_slot=0, condition='off', task_sha256='t', selection_digest='s',
                raw_float32_sha256='r', goal_sha256='g', eligible=True,
                predicted192=np.full(192, 3, np.float32).tolist(), actual_terminal192=actual.tolist(),
                goal192=np.zeros(192, np.float32).tolist(),
                original_root_context=dict(latent_history=root.tolist()))
    row = dict(spec, status='predicted', actual_full3='context', prediction='prediction')
    accepted = dict(imagined_final_Q=9., conditioned_final_Q=1., actual_final_Q=1.,
                    imagined_endpoint_MSE=4., conditioned_endpoint_MSE=0.,
                    conditioned_final192=actual.tolist())
    arrays = dict(context=mid[None], prediction=pred)
    return spec, row, dict(latent_history=mid.tolist()), accepted, arrays


def test_independent_array_comparison_and_matched_old2():
    spec, row, mid, accepted, arrays = fixture()
    r = m.compare_row(spec, row, mid, accepted, arrays.__getitem__)
    assert r['endpoint_MSE_delta'] == -4
    assert r['history_control']['matched_old2_exact']
    assert not r['ranking_repair_claim'] and not r['physical_action_changed']


@pytest.mark.parametrize('bad', ['ineligible', 'status', 'task', 'shape', 'Q', 'latent', 'history'])
def test_actual_join_or_numeric_mismatch_refused(bad):
    spec, row, mid, accepted, arrays = deepcopy(fixture())
    if bad == 'ineligible':
        spec['eligible'] = False
    elif bad == 'status':
        row['status'] = 'not_started'
    elif bad == 'task':
        row['task_sha256'] = 'other'
    elif bad == 'shape':
        arrays['prediction'] = arrays['prediction'][:, -1]
    elif bad == 'Q':
        accepted['actual_final_Q'] += 1
    elif bad == 'latent':
        accepted['conditioned_final192'][0] += 1
    else:
        mid['latent_history'][0][0] += 1
    with pytest.raises(ValueError):
        m.compare_row(spec, row, mid, accepted, arrays.__getitem__)
