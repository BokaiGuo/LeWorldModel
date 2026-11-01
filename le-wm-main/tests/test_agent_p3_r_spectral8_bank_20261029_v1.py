"""Saved ABI / NumPy fixtures only. No model, weights or production data."""
from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_r_spectral8_bank_20261029_v1 as m


def task():
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    t=dict(root_context=c,goal=[1.]+[0.]*191,projection=np.eye(192).tolist(),
        action_mean=[.125,-.25],action_std=[2.,4.])
    t['binding']={k:m.digest(k) for k in m.core.BINDINGS}
    t['binding'].update(context_sha256=m.digest(c),goal_sha256=m.digest(t['goal']),
        projection_sha256=m.digest(t['projection']),normalizer_sha256=m.digest([t['action_mean'],t['action_std']]),
        action_domain_sha256=m.digest(m.core.action_domain()))
    return m.core.validate_task(t)


def rehash(mapping):
    mapping['mapping_sha256']=m.digest({k:v for k,v in mapping.items() if k!='mapping_sha256'})


def fixture(h=1,child=False):
    root=task();actions=[];path=[];c=deepcopy(root['root_context'])
    if child:
        actions=np.full((5,2),.25,np.float32).tolist()
        token=((np.asarray(actions,np.float32)-np.asarray(root['action_mean'],np.float32))/
            np.asarray(root['action_std'],np.float32)).reshape(10)
        c=dict(c,latent_history=c['latent_history'][1:]+[[.5]+[0.]*191],
            action_history=[c['action_history'][-1],token.tolist()])
        path=[c]
    t=m.parent.parent_task(root,c)
    req=m.core.request(t,dict(context=c),operation='compile_finite_horizon_J',horizon=h)
    j=np.zeros((192,10*h),np.float32)
    # Nontrivial, nondegenerate raw J; task std != 1 catches re-scaling.
    for i in range(10*h):
        j[i,i]=np.float32((10*h-i)/8)
        j[i,(i+3)%(10*h)]=np.float32(.015625*(i+1))
    out=dict(binding_sha256=m.digest(t['binding']),parent_context_sha256=m.digest(c),horizon=h,
        nominal=c['latent_history'][-1],jacobian=j.tolist(),producer_sha256=m.core.sha(m.signal.__file__))
    mapping=dict(schema=m.parent.NAME+'.signal_map',root_task_sha256=m.digest(root),root_binding=root['binding'],
        parent_task=t,outer_request=m.core.request(root,dict(context=c),operation='compile_finite_horizon_J',horizon=h),
        native_request=req,native_response=out,outward_response=dict(out,binding_sha256=m.digest(root['binding']),
            producer_sha256=m.core.sha(m.parent.__file__)))
    rehash(mapping)
    origin=dict(kind='origin',phase='nominal',step=None,arrays={k:m.signal.blob(v) for k,v in dict(
        latent=np.asarray(c['latent_history'],np.float32)[None],history=np.asarray(c['action_history'],np.float32)[None],
        raw_flat=np.zeros(10*h,np.float32),mean=np.asarray(t['action_mean'],np.float32),std=np.asarray(t['action_std'],np.float32)).items()})
    call=[dict(kind='native_call_attempt',method='signal',task=t,request=req),origin,
        dict(kind='nominal',arrays=dict(value=m.signal.blob(np.asarray(out['nominal'],np.float32)))),
        dict(kind='jacobian',arrays=dict(value=m.signal.blob(j))),dict(kind='native_call_return',result=out)]
    outer=m.core.Ledger(dict.fromkeys(m.core.AXES,1000));outer.scope=dict(fixture=True)
    outer.call('signal_compile',lambda:None)
    evidence=dict(kind='RAM_FIXTURE',artifact_path='/RAM/explicit_signal_fixture.json',artifact_sha256=m.digest(call),
        call_id=0,native_call=call,origin_index=1,nominal_index=2,jacobian_index=3,
        signal_compile_entry=deepcopy(outer.events[0]))
    ledger=m.core.Ledger(m.bank_cost());ledger.scope=dict(bank_fixture=True)
    events=[]
    return dict(root=root,prefix_actions=actions,prefix_path=path,mapping=mapping,evidence=evidence,
        ledger=ledger,verification_reserve=dict.fromkeys(m.core.AXES,0),sink=events.append,permit_ram=True),events


def test_h1_original_last8_params_kinds_and_raw_bits_exact():
    k,events=fixture()
    r=m.build(**k)
    assert r['status']=='complete'
    native=k['mapping']['native_response']
    original=m.kernel.original.spectral_bank(native['jacobian'],native['nominal'],k['root']['goal'],k['root']['projection'])
    a=np.asarray(r['params'],np.float64);b=np.asarray(original['params'][2:],np.float64)
    assert a.tobytes()==b.tobytes()
    assert r['kinds']==original['kinds'][2:]
    assert np.asarray(r['right_vectors'],np.float64).tobytes()==np.asarray(original['right_vectors'],np.float64).tobytes()
    assert r['singular_values']==original['singular_values']
    assert np.asarray(r['actions'],np.float32).tobytes()==np.tanh(b).astype(np.float32).reshape(8,5,2).tobytes()
    assert r['events']==events and r['zero']==[[0.,0.]]*5
    assert [x['resource'] for x in r['ledger_events']]==['program_compile','array_svd']
    assert r['resources']==m.bank_cost() and r['resources']['solve_calls']==0
    assert r['completed']==r['resources']
    assert r['authority_manifest']['bank_ledger_events_sha256']==m.digest(r['ledger_events'])
    assert r['source']['resources'] is None and not r['source']['native_verified']
    assert r['authority']['bank_authenticated'] is False and r['authority']['authentication_sha256'] is None
    assert not r['policy_tree_executed'] and not r['original_R_equivalence']
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('h',[1,3])
def test_all_eight_sign_mode_radius_order_h3_same_total_radius(h):
    k,_=fixture(h,child=True)
    r=m.build(**k)
    assert r['status']=='complete' and len(r['right_vectors'])==10*h
    assert np.asarray(r['right_vectors']).shape==(10*h,10*h)
    assert np.asarray(r['actions']).shape==(8,5*h,2)
    expected=[(radius,mode,sign) for radius in m.kernel.original.RADII for mode in range(2) for sign in (-1,1)]
    assert [(x['radius'],x['mode'],x['sign']) for x in r['directions']]==expected
    for i,(radius,mode,sign) in enumerate(expected):
        assert np.array_equal(np.asarray(r['params'][i]),sign*radius*np.asarray(r['right_vectors'][mode]))
        assert np.isclose(np.linalg.norm(r['params'][i]),radius,rtol=2e-15)
    for row in r['right_vectors']:
        assert row[np.argmax(np.abs(row))]>=0
    assert r['mapping']==k['mapping'] # original J/nominal preserved; no std scaling
    assert r['projected_J_float64']==k['mapping']['native_response']['jacobian']
    # Bank authority can feed the frozen RAM kernel's exact identity validator.
    m.kernel.validate_inputs(k['root'],r['parent_task']['root_context'],k['prefix_actions'],k['prefix_path'],
        r['actions'],r['zero'],h,r['authority'])


def test_h3_explicit_extension_not_old_unsupported_call():
    k,_=fixture(3)
    with pytest.raises(ValueError,match='constant2 or full10'):
        m.kernel.original.spectral_bank(k['mapping']['native_response']['jacobian'],[0.]*192,[0.]*192,np.eye(192))
    r=m.build(**k)
    assert r['status']=='complete' and len(r['params'][0])==30
    assert r['resources']['array_SVD_calls']==1 and r['resources']['predict_API']==0


def test_equal_absolute_components_first_coordinate_sign_and_legacy_bits():
    k,_=fixture()
    j=np.zeros((192,10),np.float32);j[0,:2]=[2.,2.];j[1,:2]=[1.,-1.]
    out=k['mapping']['native_response'];out['jacobian']=j.tolist()
    k['mapping']['outward_response']['jacobian']=j.tolist()
    k['evidence']['native_call'][-1]['result']['jacobian']=j.tolist()
    k['evidence']['native_call'][3]['arrays']['value']=m.signal.blob(j)
    rehash(k['mapping'])
    r=m.build(**k)
    assert r['status']=='complete'
    for v in r['right_vectors']:assert v[np.argmax(np.abs(v))]>=0
    old=m.kernel.original.spectral_bank(j,out['nominal'],k['root']['goal'],np.eye(192))
    assert np.asarray(r['params'],np.float64).tobytes()==np.asarray(old['params'][2:],np.float64).tobytes()


def test_consumed_inputs_not_mutated_by_sign_normalization():
    k,_=fixture(3,child=True)
    before=m.digest(dict(root=k['root'],mapping=k['mapping'],evidence=k['evidence'],path=k['prefix_path']))
    r=m.build(**k)
    assert r['status']=='complete'
    assert m.digest(dict(root=k['root'],mapping=k['mapping'],evidence=k['evidence'],path=k['prefix_path']))==before
    signed=np.asarray(r['right_vectors']);original=np.asarray(r['right_vectors_original'])
    assert np.array_equal(np.abs(signed),np.abs(original))


@pytest.mark.parametrize('axis',m.core.AXES)
def test_atomic_all_axis_reserve_no_svd_or_fee(axis):
    k,events=fixture();k['verification_reserve'][axis]=1
    r=m.build(**k)
    assert r['status']=='resource_cap_before_bank'
    assert k['ledger'].events==[] and not any(r['resources'].values())
    assert r['actions'] is None and r['authority'] is None and len(events)==1


def test_used_and_verification_reserve_exact_fit():
    k,_=fixture();cost=m.bank_cost();cost['solve_calls']=3;cost['array_SVD_calls']+=1
    k['ledger']=m.core.Ledger(cost);k['ledger'].scope=dict(fixture=True)
    k['ledger'].call('solve',lambda:None)
    k['verification_reserve']['solve_calls']=2;k['verification_reserve']['array_SVD_calls']=1
    r=m.build(**k)
    assert r['status']=='complete' and r['resources']==m.bank_cost()
    assert k['ledger'].attempted['solve_calls']==1 # existing fee, not refunded or charged again


@pytest.mark.parametrize('field',['mapping_sha256','root_task_sha256','parent_task','native_request','outward_response','native_response'])
def test_mapping_mutations_before_fee(field):
    k,_=fixture(child=True);mp=k['mapping']
    if field=='mapping_sha256':mp[field]=m.digest('bad')
    elif field=='root_task_sha256':mp[field]=m.digest('other-root');rehash(mp)
    elif field=='parent_task':mp[field]['root_context']['latent_history'][0][1]=1.;rehash(mp)
    elif field=='native_request':mp[field]['parent_context_sha256']=m.digest('root-not-parent');rehash(mp)
    elif field=='outward_response':mp[field]['producer_sha256']=m.digest('wrong');rehash(mp)
    else:mp[field]['producer_sha256']=m.digest('fake');rehash(mp)
    with pytest.raises(ValueError):m.build(**k)
    assert k['ledger'].events==[]


@pytest.mark.parametrize('mutation',['J','nominal','std','mean','raw','latent','history','pointer','outerfee','call','HOST','optin','h2','float32'])
def test_saved_array_identity_source_and_paid_metadata_refusal(mutation):
    k,_=fixture();e=k['evidence'];mp=k['mapping']
    if mutation in ('std','mean','raw','latent','history'):
        name='raw_flat' if mutation=='raw' else mutation
        b=e['native_call'][1]['arrays'][name]
        a=m.signal.unblob(b).copy();a.flat[0]+=np.float32(.25)
        e['native_call'][1]['arrays'][name]=m.signal.blob(a)
    if mutation=='J':e['native_call'][3]['arrays']['value']=m.signal.blob(np.zeros((192,10),np.float32))
    if mutation=='nominal':e['native_call'][2]['arrays']['value']=m.signal.blob(np.ones(192,np.float32))
    if mutation=='pointer':e['jacobian_index']=2
    if mutation=='outerfee':e['signal_compile_entry']['delta']['signal_compile_calls']=0
    if mutation=='call':e['native_call'][0]['method']='replay'
    if mutation=='HOST':e['kind']='HOST_CUDA'
    if mutation=='optin':k['permit_ram']=False
    if mutation=='h2':mp['native_response']['horizon']=2;rehash(mp)
    if mutation=='float32':mp['native_response']['jacobian'][0][0]=.1;rehash(mp)
    with pytest.raises(ValueError):m.build(**k)
    assert k['ledger'].events==[]


def test_normalizer_must_not_rescale_parent_J_again():
    k,_=fixture();r=m.build(**k)
    original=np.asarray(k['mapping']['native_response']['jacobian'],np.float64)
    double=original/np.tile(np.asarray(k['root']['action_std']),5)
    assert not np.array_equal(double,np.asarray(r['projected_J_float64']))
    assert np.array_equal(original,np.asarray(r['projected_J_float64']))


def test_nonidentity_Q_not_relabelled_as_same_task():
    k,_=fixture();k['root']['projection'][0][0]=2.
    k['root']['binding']['projection_sha256']=m.digest(k['root']['projection'])
    with pytest.raises(ValueError,match='identity192'):m.build(**k)


def test_failed_sink_no_fee_no_fabricated_success():
    k,_=fixture();events=[]
    def sink(e):
        events.append(e)
        if e['payload']['kind']=='bank_attempt':raise RuntimeError('fixture journal error')
    k['sink']=sink
    r=m.build(**k)
    assert r['status']=='partial_failure' and r['authority'] is None
    assert not any(r['resources'].values()) and r['ledger_events']==[]


def test_missing_unknown_axis_not_zero_filled():
    k,_=fixture();k['verification_reserve'].pop('raw_backward_callbacks')
    with pytest.raises(ValueError):m.build(**k)


def test_cold_contract_no_neural_import_or_production_execution():
    code='''
import sys
from experiments.spectral_reachability import agent_p3_r_spectral8_bank_20261029_v1 as m
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
m.check_sources()
assert not m.contract()['policy_tree_executed']
'''
    subprocess.run([sys.executable,'-c',code],check=True)
    p=subprocess.run([sys.executable,'-m',m.__name__,'--contract'],capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)['candidates']==8
