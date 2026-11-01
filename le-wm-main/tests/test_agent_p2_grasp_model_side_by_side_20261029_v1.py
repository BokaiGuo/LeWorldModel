"""Saved metadata + RAM mutations only. No torch/NN/physical outcomes."""
from copy import deepcopy
import json
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_grasp_model_side_by_side_20261029_v1 as s


@pytest.fixture(scope='module')
def saved():
    d={k:s.load_pinned(s.ROOT/p,h) for k,(p,h) in s.PINS.items()}
    tasks=[s.inputs.parse(s.inputs.read_bound(x['task_file'],x['task_sha256'],compressed=True))['task'] for x in d['grasp_prepare']['slots']]
    return d,tasks


def resign(plan,producer):
    plan.pop('selection_digest');plan['selection_digest']=producer(plan)


def test_actual_six_model_plans(saved):
    d,t=saved;r=s.compare(d,t)
    assert [(x['root_slot'],x['method']) for x in r['rows']]==[
        (0,'history_aware_joint30_GRASP_variant'),(0,'joint'),(0,'composition'),
        (1,'history_aware_joint30_GRASP_variant'),(1,'composition'),(1,'joint')]
    assert [x['recomputed_model_Q'] for x in r['rows']]==[
        .13271686313507314,.3634363508809801,.32377574079538207,
        .04535450828709742,.08947335474243788,.09379228960727053]
    assert [x['resources_returned']['predict_API'] for x in r['rows']]==[2706,1902,1062,2706,1062,1902]
    assert [x['resources_returned']['scalar_backward'] for x in r['rows']]==[900,0,0,900,0,0]
    assert all(x['total_FLOPs'] is None and x['engine_internal_backward_calls'] is None for x in r['rows'])
    assert all(x['identity_domain_equal'] and x['blind_pointer']['opened'] is False for x in r['identities'])
    assert r['matched'] is r['equal_budget'] is r['equal_FLOPs'] is False
    assert r['new_neural_calls']==r['new_physics_calls']==0
    assert r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False


@pytest.mark.parametrize('kind',[
    'missing_grasp','missing_common','duplicate_root','wrong_order','task_sha','condition',
    'context','goal','Q','normalizer','model','warmup','future_horizon','blind',
    'receipt_report','report_model','report_errors','admission','grasp_resource',
    'resource_missing','resource_undercharge','resource_return_missing','cap_drift',
    'account_binding','missing_context','digest','coherent_wrong_Q','coherent_wrong_raw',
    'outside_ball','missing_logit','partial_account','physics_selection'])
def test_saved_metadata_mutation_rejected(saved,kind):
    d,t=deepcopy(saved);gp=d['grasp_prepare'];wp=d['common_prepare'];g=d['grasp_receipt'];w=d['common_receipt']
    p=w['plans'][0];a=w['accounts'][0]
    if kind=='missing_grasp':g['plans'].pop()
    elif kind=='missing_common':w['plans'].pop()
    elif kind=='duplicate_root':g['plans'][1]['root_slot']=0
    elif kind=='wrong_order':wp['root_arm_order'].reverse()
    elif kind=='task_sha':wp['slots'][0]['task_sha256']='0'*64
    elif kind=='condition':gp['slots'][0]['condition']='on'
    elif kind=='context':gp['slots'][0]['model_input']['root_context']['latent_history'][0][0]+=.01
    elif kind=='goal':p['goal'][0]+=.01;resign(p,s.wall.old.digest)
    elif kind=='Q':wp['slots'][0]['q'][0][0]=2
    elif kind=='normalizer':g['plans'][0]['normalization']['std'][0]+=.01
    elif kind=='model':gp['expected_model_hash']='0'*64
    elif kind=='warmup':wp['slots'][0]['warm_raw10'][0][0]=0
    elif kind=='future_horizon':gp['slots'][0]['target_future_packets']=2
    elif kind=='blind':gp['slots'][0]['blind_pointer']['sha256']='0'*64
    elif kind=='receipt_report':g['planner_report_sha256']='0'*64
    elif kind=='report_model':d['common_report']['model_after']='0'*64
    elif kind=='report_errors':d['grasp_report']['errors']=['failure']
    elif kind=='admission':g['P1_admission']=True
    elif kind=='grasp_resource':g['resources']['scalar_backward']=0
    elif kind=='resource_missing':a['attempted'].pop('jacobian')
    elif kind=='resource_undercharge':a['attempted']['predict']=a['completed']['predict']=0
    elif kind=='resource_return_missing':a['completed'].pop('predict')
    elif kind=='cap_drift':a['caps']['predict']=9999
    elif kind=='account_binding':a['binding']['condition']='on'
    elif kind=='missing_context':p['contexts'].pop();resign(p,s.wall.old.digest)
    elif kind=='digest':p['selection_digest']='0'*64
    elif kind=='coherent_wrong_Q':
        p['model_Q']=a['incumbent']['cost']=.123;resign(p,s.wall.old.digest)
    elif kind=='coherent_wrong_raw':
        p['commands'][0][0]=0.;p['raw_float32_bits']=np.asarray(p['commands'],'<f4').tobytes().hex();resign(p,s.wall.old.digest)
    elif kind=='outside_ball':
        g['plans'][0]['logits']=[100.]*30;resign(g['plans'][0],s.grasp.digest)
    elif kind=='missing_logit':a['incumbent']['witness'].pop('logits')
    elif kind=='partial_account':a['stop']='budget_stop'
    elif kind=='physics_selection':p['selection_uses_physics']=True;resign(p,s.wall.old.digest)
    with pytest.raises((ValueError,KeyError)):s.compare(d,t)


def test_chart_domain_and_float32_exact():
    ctx=dict(context_length=3,latent_dim=192,action_dim=10,layout='states_with_between_state_actions_v1',
             latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    request=dict(root_context=ctx,goal=np.zeros(192).tolist(),q=np.eye(192).tolist(),normalization=s.inputs.normalization())
    logits=np.linspace(-.1,.1,30);commands=np.tanh(logits).astype(np.float32).reshape(15,2).tolist()
    assert s.checked_actions(request,logits,commands)['logit_L2']<.75*np.sqrt(15)
    commands[0][0]=float(np.nextafter(np.float32(commands[0][0]),np.float32(np.inf)))
    with pytest.raises(ValueError,match='literal tanh'):s.checked_actions(request,logits,commands)


def test_cost_not_rounding_or_parity_tolerance():
    ctx=dict(context_length=3,latent_dim=192,action_dim=10,layout='states_with_between_state_actions_v1',
             latent_history=np.ones((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    assert s.checked_q(ctx,np.zeros(192).tolist(),1.)==1.
    with pytest.raises(ValueError):s.checked_q(ctx,np.zeros(192).tolist(),float(np.nextafter(1.,2.)))


def test_bounded_parser(tmp_path):
    p=tmp_path/'d.json';p.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):s.load_pinned(p,s.inputs.sha(p.read_bytes()))
    p.write_bytes(b'{"a":NaN}')
    with pytest.raises(ValueError):s.load_pinned(p,s.inputs.sha(p.read_bytes()))
    with pytest.raises(ValueError):s.load_pinned(p,'0'*64)
    p.write_bytes(b'{}');link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):s.load_pinned(link,s.inputs.sha(b'{}'))


def test_no_cross_producer_digest_substitution(saved):
    p=deepcopy(saved[0]['common_receipt']['plans'][0]);q=deepcopy(p);q.pop('selection_digest')
    # Default-space JSON is NOT this producer's compact-newline authority.
    p['selection_digest']=s.inputs.sha(json.dumps(q,sort_keys=True).encode())
    with pytest.raises(ValueError):s.authenticated(p,s.wall.old.digest)


def test_build_only_model_metadata_no_expensive_consume_or_blind(saved,monkeypatch):
    for m in (s.grasp,s.wall):
        monkeypatch.setattr(m,'consume',lambda *a,**k:pytest.fail('must reuse sealed receipt, not rerun consume'))
    from pathlib import Path
    original=Path.open
    opened=[]
    def guarded(p,*a,**k):
        p=Path(p);opened.append(str(p))
        assert p.suffix not in ('.npz','.npy','.png','.pt','.pth')
        # Bound source text can contain "physical" in its filename; forbid
        # physical RESULT reads, not inherited normalization source hashing.
        assert not ('results' in p.parts and '_physical' in str(p))
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    result=s.build()
    assert len(result['rows'])==6 and result['bindings'] and len(result['summary_bundle'])==3
    assert all(not ('/results/' in x and '_physical' in x) for x in opened)


def test_cold_import_no_neural_or_foreign():
    code="""
import sys
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tensorflow','transformers','sklearn','pymunk','gym','gymnasium'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from experiments.spectral_reachability import agent_p2_grasp_model_side_by_side_20261029_v1 as s
assert not hasattr(s,'run') and not hasattr(s,'run_cuda')
"""
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=s.ROOT,capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
