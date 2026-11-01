"""Pure vector arithmetic and synthetic target-tracking callbacks, no NN."""
from copy import deepcopy
import hashlib
import json
import math
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p2_direct_latent_optimizer_20261029_v1 as m


def task():
    context=dict(layout='RAM_explicit_full3',context_length=3,latent_dim=192,action_dim=10,
        latent_history=[[float(j)]*192 for j in (1,2,3)],action_history=[[0.]*10,[.1]*10])
    return dict(schema=m.NAME+'.task',root_slot=0,condition='off',context=context,goal=[4.]*192,
        coordinate_refs={k:dict(path='/fixture/'+k+'.json',sha256='a'*64) for k in m.COMMON})


def controller(t):
    budget=dict(mode='predict_API_calls_matched',cap=10,matched_model_sha256='b'*64,
        matched_callable_sha256='d'*64,batch_limit=1,scope=list(m.PHASES),failed_attempts_refunded=False)
    return dict(source=dict(path='/fixture/tracker.py',sha256='c'*64),
        predictor_source=dict(path='/fixture/predictor.py',sha256='d'*64),
        model_hash='b'*64,coordinate_refs=deepcopy(t['coordinate_refs']),budget=budget,
        same_callable_all_arms=True,privileged_compiled_warmstart=False)


def test_fixed_64_real_direct_latent_updates_full_history():
    t=task();before=deepcopy(t);r=m.optimize(t)
    assert t==before and len(r['history'])==65 and r['history'][0]['targets']==[[3.]*192]*3
    assert all(row['score']['loss']>=nxt['score']['loss'] for row,nxt in zip(r['history'],r['history'][1:]))
    assert r['history'][-1]['gradient'] is None
    assert all(row['gradient'] is not None for row in r['history'][:-1])
    assert r['delivered_targets']==r['history'][-1]['targets']
    assert r['delivered_targets']!=[t['goal']]*3
    assert r['delivered_targets']!=t['context']['latent_history']
    # Analytic optimizer of THIS declared convex surrogate, not model dynamics.
    for k,z in enumerate(r['delivered_targets']):assert max(abs(x-(3+(k+1)/4)) for x in z)<.002
    assert r['contract']['uses_action_rollout_as_target'] is False and not r['dispatch_allowed']
    assert m.raw(r)==m.raw(m.optimize(t))


def test_zero_direction_still_all_steps_no_finalgoal_substitution():
    t=task();t['goal']=[3.]*192;r=m.optimize(t)
    assert len(r['history'])==65 and all(x['score']['loss']==0 for x in r['history'])
    assert r['delivered_targets']==[[3.]*192]*3


def test_analytic_gradient_all_576_coordinates_finite_difference():
    t=task();anchor=t['context']['latent_history'][-1];goal=t['goal']
    z=[[3.+.01*k+.0001*i for i in range(192)] for k in range(3)]
    _,g=m.objective(z,anchor,goal);h=1e-4
    for k in range(3):
        for i in range(192):
            lo=deepcopy(z);hi=deepcopy(z);lo[k][i]-=h;hi[k][i]+=h
            fd=(m.objective(hi,anchor,goal,False)[0]['loss']-m.objective(lo,anchor,goal,False)[0]['loss'])/(2*h)
            assert abs(fd-g[k][i])<1e-9


def test_eta_global_bound_not_outcome_tuning():
    # H=(2/192)*tridiag(-1,2,-1), eigenvalues(2/192)*(2-2cos(k*pi/4)).
    eig=[(2/192)*(2-2*math.cos(k*math.pi/4)) for k in (1,2,3)]
    assert m.ETA<1/max(eig) and m.ETA<192/8
    assert m.contract()['updates']==64 and m.contract()['eta']==16.
    with pytest.raises(TypeError):m.optimize(task(),steps=8)


@pytest.mark.parametrize('bad',('bool','nan','short','history','extraactions','outcome','condition','shape','learned'))
def test_task_rejects_invalid_and_privileged_inputs(bad):
    t=task()
    if bad=='bool':t['goal'][0]=True
    elif bad=='nan':t['goal'][0]=float('nan')
    elif bad=='short':t['goal'].pop()
    elif bad=='history':t['context']['action_history'].pop()
    elif bad=='extraactions':t['old_compiled_commands']=[[0.,0.]]*15
    elif bad=='outcome':t['native_success']=True
    elif bad=='condition':t['condition']='on'
    elif bad=='shape':t['context']['context_length']=True
    else:t['learned_prior_targets']=[[0.]*192]*3
    with pytest.raises(ValueError):m.optimize(t)


def test_actual_actions_do_not_parameterize_the_latent_optimizer():
    t=task();r=m.optimize(t);t['context']['action_history']=[[100.]*10,[-100.]*10]
    other=m.optimize(t)
    assert r['delivered_targets']==other['delivered_targets']
    assert r['context_sha256']!=other['context_sha256']


def test_arbitrary_asset_matches_Pauli_interface_not_science():
    t=task();a=m.optimizer_asset('b'*64,t['coordinate_refs'])
    assert a['schema']==m.PROTOCOL+'.arbitrary_optimizer' and a['parameter_shape']==[3,192]
    assert a['method']=='direct_latent_optimization' and a['uses_action_rollout_as_target'] is False
    assert a['production_ready'] is False


class RAMTracker:
    kind='RAM_FIXTURE';source_sha256='c'*64
    def __init__(self):self.calls=[]
    def __call__(self,req):
        self.calls.append(req)
        return dict(predicted=deepcopy(req['delivered_targets']),observed=[None,None,None],
            resources=dict(attempted=dict(predict_API=2,predictor_forward_samples=2),
                completed=dict(predict_API=1,predictor_forward_samples=1)))


def test_unified_all_three_arms_have_same_public_request_no_witness():
    t=task();c=controller(t);targets=m.optimize(t)['delivered_targets'];tracker=RAMTracker()
    reqs=[m.tracking_request(t,arm,targets,c) for arm in m.ARMS]
    for req in reqs:
        r=m.fixture_track(req,tracker)
        assert r['real_meter_authenticated'] is False and r['other_resources']=='unknown'
        assert all(x['actual_to_delivered'] is None for x in r['metrics']['rows'])
        assert all(x['predicted_to_delivered']['MSE']==0 for x in r['metrics']['rows'])
        assert r['metrics']['native_success'] is r['metrics']['realizability_ratio'] is None
        assert req['privileged_witness_actions'] is None
    stripped=[{k:v for k,v in req.items() if k!='arm'} for req in reqs]
    assert m.raw(stripped[0])==m.raw(stripped[1])==m.raw(stripped[2])
    # Supplying the same synthetic target above tests ABI uniformity only; it
    # does NOT generate/claim learned or compiled proposals.
    assert len(tracker.calls)==3


def test_missing_prior_or_controller_not_faked():
    t=task()
    r=m.tracking_request(t,m.ARMS[1],None,controller(t))
    assert r['status']=='blocked_missing_delivered_targets' and r['request'] is None
    r=m.tracking_request(t,m.ARMS[0],[[3.]*192]*3,None)
    assert r['status']=='blocked_missing_common_controller' and not r['dispatch_allowed']


@pytest.mark.parametrize('bad',('warmstart','pairing','predictor','budget','cap','callbackkind','callbacksource'))
def test_tracker_binding_and_cap_rejection(bad):
    t=task();c=controller(t);tracker=RAMTracker()
    if bad=='warmstart':c['privileged_compiled_warmstart']=True
    elif bad=='pairing':c['coordinate_refs']['encoder']['sha256']='e'*64
    elif bad=='predictor':c['predictor_source']['sha256']='e'*64
    elif bad=='budget':c['budget']['cap']=True
    elif bad=='cap':c['budget']['cap']=1
    elif bad=='callbackkind':tracker.kind='HOST_CUDA'
    else:tracker.source_sha256='e'*64
    with pytest.raises(ValueError):
        req=m.tracking_request(t,m.ARMS[0],[[3.]*192]*3,c)
        m.fixture_track(req,tracker)


def test_tracking_is_delivered_target_not_taskgoal_and_keeps_null():
    t=task();req=m.tracking_request(t,m.ARMS[0],[[3.]*192]*3,controller(t))
    r=m.tracking_metrics(req,[[4.]*192]*3,[[3.]*192,None,[4.]*192])
    assert r['rows'][0]['predicted_to_delivered']['MSE']==1.
    assert r['rows'][0]['actual_to_delivered']['MSE']==0.
    assert r['rows'][1]['actual_to_delivered'] is None and r['checkpoint_denominator']==3


def test_saved_metadata_ref_tamper_and_reject_NPY(tmp_path):
    path=tmp_path/'small.json';path.write_text('{}');r=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert m.read_ref(r)==b'{}'
    path.write_text('{"x":1}')
    with pytest.raises(ValueError):m.read_ref(r)
    path=tmp_path/'weight.npy';path.write_bytes(b'not opened')
    with pytest.raises(ValueError):m.read_ref(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_sealed_numerical_synthetic_task_only_and_binding_drift(tmp_path):
    t=task()
    for role in m.COMMON:
        path=tmp_path/(role+'.json');path.write_text('{"fixture_only":true}')
        t['coordinate_refs'][role]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path=tmp_path/'task.json';path.write_bytes(m.raw(t));h=hashlib.sha256(path.read_bytes()).hexdigest()
    r=m.sealed_numerical(path,h)
    assert r['input_asset_AUTHENTICATION']=='SHA_only_not_encoder_pairing_or_full_strict'
    assert len(r['history'])==65 and r['dispatch_allowed'] is False
    assert all(set(ref)=={'path','sha256'} for ref in r['bindings'])
    (tmp_path/'q.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='SHA'):m.sealed_numerical(path,h)


def test_no_neural_or_physical_import_and_cli_contract():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "numpy" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,check=True)
    result=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'contract'],
        cwd=m.ROOT,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['original_protocol_exact_loss_specified'] is False
