"""CPU-only negative controls; does not change the sealed HOST test/source."""
from copy import deepcopy
from pathlib import Path
import runpy
import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_gradient_solver_20261029_v1 as solver
from experiments.spectral_reachability import p2_h15_gradient_matched_host_20261029_v1 as host
from experiments.spectral_reachability import p2_h15_gn_direction_guard_20261029_v1 as fd


def test_objective_gradient_matches_difference_and_first_four_updates():
    matrix=np.zeros((192,30));matrix[:30]=np.diag(np.linspace(.3,1.5,30))
    goal=np.linspace(-.3,.4,192);seen=[]
    def evaluate(raw):
        seen.append(np.asarray(raw).copy())
        return matrix@np.asarray(raw,np.float64).reshape(30)
    def linearize(z):
        return matrix@solver.raw(z).reshape(30), matrix*(1-np.tanh(z)**2)[None,:]
    result=solver.solve(goal,evaluate,linearize)
    initial=np.zeros(30);gradient=(2/192)*matrix.T@(-goal)
    direction=np.linspace(-1.,1.,30);direction/=np.linalg.norm(direction)
    # At zero, raw +/- eps is exactly float32; use a conservative float32 FD tolerance.
    eps=1e-3
    def objective(z):
        residual=matrix@solver.raw(z).reshape(30)-goal
        return np.mean(residual**2)
    fd_value=(objective(initial+eps*direction)-objective(initial-eps*direction))/(2*eps)
    assert fd_value==pytest.approx(float(gradient@direction),rel=1e-4,abs=1e-8)
    for i,step in enumerate(solver.STEPS):
        expected=solver.raw(solver.project(initial-step*gradient))
        np.testing.assert_array_equal(seen[i+1],expected)
    assert result['evaluation_calls']==17 and len(seen)==17


@pytest.mark.parametrize('direction', [np.eye(30)[0],np.ones(30)/np.sqrt(30)])
def test_difference_at_ball_boundary_and_wrong_sign_rejected(direction):
    z=np.zeros(30);z[0]=solver.RADIUS
    j=np.zeros((192,30));j[:30]=np.diag(1-np.tanh(z)**2)
    result=dict(logits=z.tolist(),raw=solver.raw(z).tolist(),logit_jacobian=j.tolist())
    a,b=fd.ad.probes(z,direction)
    assert np.linalg.norm(a)<=solver.RADIUS+1e-12 and np.linalg.norm(b)<=solver.RADIUS+1e-12
    def value(x):return np.pad(solver.raw(x).reshape(30).astype(np.float64),(0,162))
    fd.ad.check_difference(a,b,value(a),value(b),j)
    with pytest.raises(ValueError,match='directional difference'):
        fd.ad.check_difference(a,b,value(a),value(b),-j)
    assert np.array_equal(np.asarray(result['raw'],np.float32),solver.raw(z))


@pytest.fixture(scope='module')
def sealed_slot(tmp_path_factory):
    fixtures=runpy.run_path(str(Path(__file__).parent/'test_p2_h15_gradient_matched_host_20261029_v1.py'))
    task=deepcopy(host.bridge.build_bridge()['roots'][0]['task'])
    task['root_context']['latent_history'][-1][2]=0.0
    task['binding']['context_sha256']=host.core.digest(task['root_context'])
    path=tmp_path_factory.mktemp('gradient_negative')/'slot'
    report,sha=host.run_slot(path,task,lambda sink:fixtures['Fixture'](task,sink),lambda:None,'RAM_FIXTURE',0)
    assert report['status']=='complete',report['error']
    assert host.audit_slot(path,sha,task,'RAM_FIXTURE',0)['status']=='complete'
    return path,task,report,sha


@pytest.mark.parametrize('damage', ['result_score','budget_cap','completed_vjp','root','status'])
def test_saved_consumer_rejects_mutated_report(sealed_slot,monkeypatch,damage):
    path,task,report,sha=sealed_slot; altered=deepcopy(report)
    if damage=='result_score':altered['result']['solution']['score']+=1
    elif damage=='budget_cap':altered['resources']['caps']['predict_samples']+=1
    elif damage=='completed_vjp':altered['resources']['completed']['VJP_directions']-=1
    elif damage=='root':altered['root']=1
    else:altered['status']='failed'
    original=host.bridge.source.prep.read
    def read(filename,digest):
        if Path(filename)==path/'report.json':
            assert digest==sha
            return altered
        return original(filename,digest)
    monkeypatch.setattr(host.bridge.source.prep,'read',read)
    with pytest.raises((ValueError,AssertionError)):
        host.audit_slot(path,sha,task,'RAM_FIXTURE',0)


@pytest.mark.parametrize('damage',['basis','raw_gradient','direction_order','chain'])
def test_saved_ad_witness_negative_controls(sealed_slot,damage):
    path,_,_,_=sealed_slot
    events=[e['payload'] for e in host.life.transport.read_journal(path/'events.jsonl')]
    start=next(i for i,e in enumerate(events) if e['kind']=='native_call_attempt' and e['method']=='current_ad')
    end=next(i for i in range(start+1,len(events)) if events[i]['kind']=='native_call_return')
    call=deepcopy(events[start:end+1]);blob=host.life.native.signal
    if damage=='basis':
        e=next(x for x in call if x['kind']=='io_input' and 'grad:basis' in x['arrays'])
        a=blob.unblob(e['arrays']['grad:basis']).copy();a[0]=0;e['arrays']['grad:basis']=blob.blob(a)
    elif damage=='raw_gradient':
        e=next(x for x in call if x['kind']=='raw_backward')
        a=blob.unblob(e['arrays']['value']).copy();a[0]+=1;e['arrays']['value']=blob.blob(a)
    elif damage=='direction_order':
        e=next(x for x in call if x['kind']=='direction');e['direction']=1
    else:call[-1]['result']['logit_jacobian'][0][0]+=1
    with pytest.raises(ValueError):host.slot_impl.audit_ad(call)
