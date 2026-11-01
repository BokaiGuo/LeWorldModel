from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e1_tracking_error_decomposition_20261029_v1 as m


@pytest.fixture(scope='module')
def inputs():
    refs=m.refs();d={k:m.algebra.read(v['path'],v['sha256']) for k,v in refs.items()}
    folder=m.Path(refs['report']['path']).parent;r=d['report']
    z=[None if x['latent'] is None else m.algebra.load_latent(folder,x['latent'],r['artifact_sha256']) for x in r['rows']]
    return d['prepare'],r,d['receipt'],d['physical_receipt'],d['physical_report'],z


def test_full_saved_roster_and_pair_semantics(inputs):
    p,r,c,n,nr,z=inputs;v=m.analyze(p,r,c,n,nr,z)
    assert v['fixed_slots']==6 and len(v['rows'])==len(v['pairs'])==18
    assert all(x['cross_target_tracking_delta'] is None for x in v['pairs'])
    assert v['FLOPs'] is None and v['scientific_admission'] is False
    for x in v['rows']:
        for q in x['arithmetic']['residual'].values():
            assert abs(q['identity_roundoff_residual']) < 1e-12


@pytest.mark.parametrize('fault',['order','target','checkpoint','receipt','goal','model','resource','latent'])
def test_source_or_result_tampering_rejected(inputs,fault):
    p,r,c,n,nr,z=deepcopy(inputs)
    if fault=='order':p['rows'][0]['slot']=1
    if fault=='target':p['rows'][0]['target192'][0]+=1
    if fault=='checkpoint':p['rows'][1]['checkpoint']['state'][0]+=1
    if fault=='receipt':c['rows'][1]['model_task_Q']+=1
    if fault=='goal':p['rows'][3]['goal192'][0]+=1;r['rows'][3]['goal192']=deepcopy(p['rows'][3]['goal192'])
    if fault=='model':r['model_after']='changed'
    if fault=='resource':c['resources']['encoded_images']+=1
    if fault=='latent':z[0]=z[0].copy();z[0][0]+=np.float32(1)
    with pytest.raises(ValueError):m.analyze(p,r,c,n,nr,z)


def test_exact_decomposition_cross_and_unknown():
    p=np.ones(192,np.float32);z=np.full(192,2,np.float32);t=np.full(192,3,np.float32);g=np.zeros(192,np.float32)
    x=m.algebra.decompose(p,z,t,g)
    assert x['residual']['task_goal']['prediction_residual_energy']==1
    assert x['residual']['task_goal']['signed_residual_cross_term']==-4
    assert x['residual']['task_goal']['model_goal_MSE']==1
    assert m.algebra.decompose(p,None,t,g)['residual']['task_goal']['actual_goal_MSE'] is None
