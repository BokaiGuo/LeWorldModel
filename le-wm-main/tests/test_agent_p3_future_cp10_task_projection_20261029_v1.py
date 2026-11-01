from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_future_cp10_task_projection_20261029_v1 as m

OLD = runpy.run_path(str(Path(__file__).parent/'test_agent_p3_same_time_capability_probe_cuda_20261029_v1.py'))


@pytest.fixture
def inputs():
    bp, br, _, _, _, tasks = OLD['bridge_fixture']()
    return deepcopy(bp['rows']), [np.asarray(x['task']['root_context']['latent_history'][-1], dtype=np.float32) for x in tasks], [x['latent'] for x in br['rows']], tasks


def test_actual_task_shape_and_tokens_match_reviewed_abi(inputs):
    specs, zs, descs, old_tasks = inputs
    result = m.project(specs, zs, descs)
    assert len(result['signals']) == 7 and len(result['endpoints']) == 5
    assert not result['execution_authorized'] and not result['source_authority_verified']
    for endpoint, item in zip(result['endpoints'], old_tasks, strict=True):
        assert endpoint['task']['root_context'] == item['task']['root_context']
        assert endpoint['task']['goal'] == item['task']['goal']
        assert endpoint['identity']['namespace'] == m.NAME
        assert endpoint['task']['binding']['task_id'] != item['task']['binding']['task_id']


@pytest.mark.parametrize('damage', ['short', 'order', 'raw', 'tokens', 'time', 'snapshot', 'same_root', 'latent_shape', 'latent_dtype', 'latent_nan', 'descriptor'])
def test_reject_invalid_inputs(inputs, damage):
    specs, zs, descs, _ = inputs
    if damage == 'short': specs.pop()
    elif damage == 'order': specs.reverse()
    elif damage == 'raw': specs[0]['causal_raw10'][0][0] = 1
    elif damage == 'tokens': specs[0]['normalized_action_history'][0][0] += 1
    elif damage == 'time': specs[0]['history_sources'][0]['absolute_step'] = 9
    elif damage == 'snapshot': specs[0]['source']['snapshot']['state'][0] += 1
    elif damage == 'same_root': specs[1]['cp5_task']['goal'][0] += 1
    elif damage == 'latent_shape': zs[0] = zs[0][:191]
    elif damage == 'latent_dtype': zs[0] = zs[0].astype(np.float64)
    elif damage == 'latent_nan': zs[0][0] = np.nan
    elif damage == 'descriptor': descs[0]['shape'] = [191]
    with pytest.raises((ValueError, KeyError, AssertionError)): m.project(specs, zs, descs)
