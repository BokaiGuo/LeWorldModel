"""Non-neural arithmetic/receipt tests; never efficacy evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import pusht_feedback_suffix_compile_20261027_v1 as p


def sample():
    jac=np.zeros((192,10)); jac[:10]=np.eye(10)
    nominal=np.zeros(192); goal=np.ones(192); q=np.eye(192)
    bank=p.packet.spectral_bank(jac,nominal,goal,q)
    prefix=np.zeros((5,2)).tolist(); candidates=[]
    for i,params in enumerate(bank['params']):
        u=np.asarray(params,dtype=np.float32).astype(float)
        pred=nominal+jac@u
        candidates.append(dict(candidate_id=f'suffix_{i}',kind=bank['kinds'][i],suffix_params=u.tolist(),
            commands=deepcopy(prefix)+np.tanh(u).reshape(5,2).tolist(),predicted_terminal=pred.tolist(),
            nonlinear_terminal_cost=float(np.mean((goal-pred)**2))))
    return dict(root_sha256=p.response.ROOT,root_index=1,role='selected',arm='actual',prefix=prefix,
        context=dict(latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist()),
        q=q.tolist(),goal=goal.tolist(),jacobian=jac.tolist(),nominal_terminal=nominal.tolist(),
        candidates=candidates,selected_candidate_id=min(candidates,key=lambda c:c['nonlinear_terminal_cost'])['candidate_id'],
        independent_replay_exact=True)


def test_valid_plan():
    assert p.verify_plan(sample())['candidate_id']=='suffix_1'


@pytest.mark.parametrize('damage',['root','prefix','winner','missing','cost','scale','replay','params'])
def test_damage(damage):
    plan=deepcopy(sample())
    if damage=='root': plan['root_sha256']='0'*64
    if damage=='prefix': plan['candidates'][0]['commands'][0][0]=.1
    if damage=='winner': plan['selected_candidate_id']='suffix_0'
    if damage=='missing': plan['candidates'].pop()
    if damage=='cost': plan['candidates'][0]['nonlinear_terminal_cost']+=.1
    if damage=='scale': plan['candidates'][0]['commands'][5][0]=2
    if damage=='replay': plan['independent_replay_exact']=False
    if damage=='params': plan['candidates'][0]['suffix_params'][0]=.01
    with pytest.raises(ValueError): p.verify_plan(plan)


def test_fixed_contract():
    c=p.contract()
    assert c['predict_samples']==6*(2+10+1)==78
    assert c['jacobian_materializations']==6 and c['scalar_vjp_directions']==1152
    assert c['svd_calls']==12 and c['visual_encode_calls']==0
    assert not c['P1_admission'] and not c['total_compute_matched']
