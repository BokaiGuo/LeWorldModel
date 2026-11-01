"""Saved real targets and fixture backend only: no neural or physical execution."""
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture

from experiments.spectral_reachability import agent_e1_six_slot_common_cuda_20261029_v2 as m


def test_all_six_real_target_bytes_and_identity():
    six, previous = m.authorities()
    rows = m.fixed_rows(six)
    assert len(rows) == 6
    for row in rows:
        slot = previous['slots'][row['root_slot']]
        assert (row['task_id'],row['task_sha256'],row['blind_sha256']) == (
            slot['task_id'],slot['task_sha256'],slot['blind_sha256'])
        data = np.asarray(m.target_vector(row),dtype='<f4').tobytes()
        assert len(data)==768 and m.sha(data)==row['target_data_sha256']
        if row['arm']==m.core.ARMS[1]:
            assert row['origin']['kind']=='inherited_hi_FIRST'
            assert row['origin']['data_sha256']==row['target_data_sha256']


@pytest.mark.parametrize('root',[0,1])
def test_real_learned_target_controller_fixture(root,tmp_path):
    six,_ = m.authorities()
    row=six['rows'][root*3+1]
    target=m.target_vector(row)
    task=fixture.task(root)
    store=m.Store(tmp_path/('learned_'+str(root)))
    meter=m.phase.PhaseMeter('controller',store.event)
    dispatch=m.phase.beam.Dispatch(task,fixture.Backend(),meter,store.event)
    with nullcontext():
        result=m.core.optimize(task,root,m.core.ARMS[1],target,dispatch,store.event)
    store.put('learned.result.z',result)
    store.close()
    report=dict(artifacts={p.name:m.sha(p.read_bytes()) for p in store.path.iterdir()},
        journal_events=store.index,journal_sha256=store.previous,array_bytes=store.archive.size,metadata_bytes=store.meta)
    for name in ('report.json','completion.json','timing.json'):
        m.put_new(store.path,name,b'{}\n')
    payload, events=m.read_store(store.path,report)
    assert payload['learned.result.z']==result
    rebuilt=m.verify_controller(task,root,m.core.ARMS[1],target,result,events)
    assert rebuilt['target']==target and meter.counts()==m.phase.beam.budget(144)
    assert dispatch.index==72 and m.sha(np.asarray(target,dtype='<f4').tobytes())==row['target_data_sha256']
    with pytest.raises(ValueError):
        m.verify_controller(task,root,m.core.ARMS[1],m.target_vector(six['rows'][(1-root)*3+1]),result,events)


def test_actual_target_mutation_and_missing_slot_rejected():
    six,_=m.authorities()
    bad=deepcopy(six)
    bad['rows'][1]['target_f32_le_base64']=bad['rows'][4]['target_f32_le_base64']
    with pytest.raises(ValueError):m.fixed_rows(bad)
    bad=deepcopy(six);bad['rows'].pop()
    with pytest.raises(ValueError):m.fixed_rows(bad)
