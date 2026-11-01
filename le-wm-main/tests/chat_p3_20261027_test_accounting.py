from copy import deepcopy
from experiments.spectral_reachability import chat_p3_20261027_accounting as a
from experiments.spectral_reachability import chat_p3_20261027_cuda as cuda
import numpy as np
import pytest


def test_shape_axes_not_API_inference():
    events=[];index=0
    for group in ('shared',)+cuda.ARMS:
        for resource in cuda.old.MAXIMA:
            e=dict(index=index,resource=resource,scope=[group,0],started_seconds=1.,completed=False)
            events.extend([dict(kind='resource_attempt',entry=deepcopy(e)),dict(kind='resource_return',entry=dict(e,ended_seconds=2.,completed=True))])
            if resource in ('predictor','action_encoder','pred_proj'):
                shape=(6,192) if resource=='pred_proj' else (2,3,192)
                events.append(dict(kind='module_input',scope=[group,0],resource=resource,entry=index,arrays=[cuda.old.blob(np.zeros(shape,np.float32))]))
            index+=1
    views=a.resource_views(events)
    assert all(v['calls']['predict']['completed']==1 and v['predictor_batch_samples']==2 and v['predictor_context_tokens']==6 and v['pred_proj_flattened_rows']==6 for v in views.values())
    assert all(v['resource_inclusive_wall_seconds']['predict']==1. for v in views.values())


def test_manipulation_must_use_actual_prefix_not_allocation():
    panel={'rows':[]};physical={'rows':[]}
    for slot in (0,1):
        for arm,n in zip(cuda.ARMS,(30,45,15,30)):
            plan=dict(root_sha256=str(slot),task_id=str(slot),blind_sha256='x',task_export='x',task_sha256='x',task_contract_sha256='x',bank_sha256='x',commands=[[0]],model_cost=1.)
            panel['rows'].append(dict(source_slot=slot,arm=arm,candidates=[dict(index=i) for i in range(n)],winner_index=0,plan=plan))
            physical['rows'].append(dict(root_slot=slot,arm=arm,initial_exact=True,goal=[0],root_state=[1],final_state=[2],initial_metrics={},final_metrics={}))
    roots=a.manipulation(panel,physical)
    assert all(r['three_distinct_prefixes'] and r['all_literal_commands_equal'] for r in roots)
    panel['rows'][1]['candidates']=deepcopy(panel['rows'][0]['candidates'])
    assert not a.manipulation(panel,physical)[0]['three_distinct_prefixes']
    panel['rows'][1]['plan']['task_sha256']='different'
    with pytest.raises(ValueError):a.manipulation(panel,physical)
