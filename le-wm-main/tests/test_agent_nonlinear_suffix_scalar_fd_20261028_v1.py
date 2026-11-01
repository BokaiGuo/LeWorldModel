"""Scalar finite differences: RAM/native ledger tests only, no neural execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_nonlinear_suffix_scalar_fd_20261028_v1 as m


def fixture_backend(h,p,sink,ledger):
    class RAM(h.RAM):
        def objective(self,after):
            y=np.asarray(after['latent_history'][-1],np.float32);g=np.asarray(self.task['goal'],np.float32)
            d=y-g;s=d*d
            return {k:m.signal.blob(v) for k,v in dict(endpoint=y,goal=g,residual=d,square=s,
                loss=np.asarray([s.mean()],np.float32)).items()}
    return RAM(p['task'],sink,ledger)


def point():
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())
    return dict(index=0,prefix_index=7,context=c,context_sha256=m.digest(c),raw5=np.zeros((5,2),np.float32).tolist(),
        raw_gradient=np.ones((5,2),np.float32).tolist(),goal=np.zeros(192,np.float32).tolist(),
        anchor_native_float32_loss=0.,task=dict(root_context=deepcopy(c),goal=np.zeros(192,np.float32).tolist(),
            action_mean=[0.,0.],action_std=[1.,1.]))


def test_fixed240_full3steps_ten_coords_both_signs():
    p=point();rs=m.requests(p)
    assert len(rs)==60 and len(rs)*4==240
    assert [(r['h'],r['coordinate'],r['sign']) for r in rs]==[(h,c,s) for h in (.01,.001,.0001) for c in range(10) for s in (-1,1)]
    for r in rs:
        a=np.asarray(r['commands'],np.float32).reshape(10)
        assert np.count_nonzero(a)==1 and r['legal']
        assert a[r['coordinate']]==np.float32(r['sign']*r['h'])
        assert r['actual_delta']==float(a[r['coordinate']])


def test_domain_missing_retained_not_clamped():
    p=point();p['raw5'][0][0]=1.;rs=m.requests(p)
    assert len(rs)==60
    for k in (1,21,41):
        assert not rs[k]['legal'] and rs[k]['missing_reason']=='outside_raw_cube'
        assert rs[k]['commands'][0][0]>1.


def test_float32_collapsed_probe_has_explicit_missing():
    p=point()
    # Test arithmetic with a separately tiny fixed h, never a production option.
    r=m.perturb(p['raw5'],0,1,1e-50,0)
    assert not r['legal'] and r['missing_reason']=='float32_step_collapsed'


def test_no_cast_repairs_or_signed_zero_changes():
    with pytest.raises(ValueError):m.packet([[.1,.2]]*5)
    p=point();p['raw5'][4][1]=-0.
    for r in m.requests(p):
        if r['coordinate']!=9:assert np.signbit(np.asarray(r['commands'],np.float32)[4,1])


def test_primary_nativeQ_and_actual_spacing_are_separate_no_pass_threshold():
    p=point();rs=m.requests(p);values=[np.float32(r['sign']*r['h']).item() for r in rs]
    out=m.summarize(p,rs,values)
    assert len(out['steps'])==3 and out['pair_denominator']==30
    for s in out['steps']:
        assert len(s['pairs'])==10 and s['complete']
        assert s['pairs'][0]['nominal_FD']==(values[s['step_index']*20+1]-values[s['step_index']*20])/(2*s['h'])
        assert s['pairs'][0]['actual_spacing_FD']==1.
    assert out['AD_correctness_certified'] is False


def test_missing_one_side_blocks_whole_vector_metrics_but_keeps_all_pairs():
    p=point();rs=m.requests(p);vs=[1.]*60;vs[0]=None
    out=m.summarize(p,rs,vs)
    assert out['steps'][0]['metrics_nominal'] is None and not out['steps'][0]['complete']
    assert out['steps'][0]['pairs'][0]['nominal_FD'] is None
    assert len(out['steps'][0]['pairs'])==10


def test_objective_exact_native_buffers_and_mean_rounding_diagnostic():
    y=np.arange(192,dtype=np.float32)/np.float32(193);g=np.zeros(192,np.float32)
    d=y-g;s=d**2;q=np.asarray([s.mean()],np.float32)
    arrays={k:m.signal.blob(v) for k,v in dict(endpoint=y,goal=g,residual=d,square=s,loss=q).items()}
    out=m.check_objective(arrays,y,g)
    assert out['native_float32_Q']==float(q[0]) and out['float64_endpoint_Q']>=0
    arrays['residual']=m.signal.blob(d+1)
    with pytest.raises(ValueError):m.check_objective(arrays,y,g)


def test_full_native_schedule_uses_original_forward_ledger_without_AD():
    spec=importlib.util.spec_from_file_location('scalar_fd_RAM',Path(__file__).with_name('test_agent_p1_nonlinear_suffix_library_20261028_v2.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    p=point();rs=m.requests(p);sink=h.Events();ledger=m.src.core.Ledger(m.src.caps(60,0),sink=sink.event)
    ram=fixture_backend(h,p,sink,ledger);row=m.new_row(0)
    m.evaluate(p,rs,ram,row,lambda e:None)
    m.validate_point(p,rs,row,ledger.receipt(),sink.events)
    assert row['status']=='recorded' and len(row['evaluations'])==60
    assert ledger.completed['grad_API']==0 and ledger.completed['predict_API']==60


@pytest.mark.parametrize('kind',['unknown','order','context','raw','objective','receipt'])
def test_full_native_mutations_rejected(kind):
    spec=importlib.util.spec_from_file_location('scalar_fd_RAM_mut',Path(__file__).with_name('test_agent_p1_nonlinear_suffix_library_20261028_v2.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    p=point();rs=m.requests(p);sink=h.Events();ledger=m.src.core.Ledger(m.src.caps(60,0),sink=sink.event)
    ram=fixture_backend(h,p,sink,ledger);row=m.new_row(0);m.evaluate(p,rs,ram,row,lambda e:None)
    receipt=ledger.receipt()
    if kind=='unknown':sink.events.append(dict(kind='unreleased'))
    elif kind=='order':sink.events[0],sink.events[1]=sink.events[1],sink.events[0]
    elif kind=='context':row['evaluations'][0]['context']['latent_history'][0][0]=1.
    elif kind=='raw':row['evaluations'][0]['request']['commands'][0][0]=0.
    elif kind=='objective':row['evaluations'][0]['objective']['arrays']['loss']=m.signal.blob(np.array([100.],np.float32))
    else:receipt['completed']['grad_API']=1
    with pytest.raises((ValueError,KeyError)):m.validate_point(p,rs,row,receipt,sink.events)


@pytest.mark.parametrize('target',['context','packet'])
def test_signed_zero_native_context_is_not_numeric_equality(target):
    spec=importlib.util.spec_from_file_location('scalar_fd_RAM_zero',Path(__file__).with_name('test_agent_p1_nonlinear_suffix_library_20261028_v2.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    p=point();rs=m.requests(p);sink=h.Events();ledger=m.src.core.Ledger(m.src.caps(60,0),sink=sink.event)
    row=m.new_row(0);m.evaluate(p,rs,fixture_backend(h,p,sink,ledger),row,lambda e:None)
    if target=='context':
        first=next(e for e in sink.events if e['kind']=='forward_started')
        first['input']['latent_history'][0][0]=-0.
    else:
        first=next(e for e in sink.events if e['kind']=='program_packet')
        a=m.signal.unblob(first['arrays']['latent']).copy();a[0,0,0]=-0.
        first['arrays']['latent']=m.signal.blob(a)
    with pytest.raises(ValueError):m.validate_point(p,rs,row,ledger.receipt(),sink.events)


def test_partial_forward_has_durable_intent_and_all60_slots_no_retry():
    class Fail:
        def step(self,*args):raise RuntimeError('injected forward failure')
    p=point();row=m.new_row(0);events=[]
    with pytest.raises(RuntimeError,match='injected'):m.evaluate(p,m.requests(p),Fail(),row,events.append)
    assert len(row['evaluations'])==60 and row['status']=='running'
    assert row['evaluations'][0]['status']=='started'
    assert all(x['status']=='not_started' for x in row['evaluations'][1:])
    assert [e['kind'] for e in events]==['probe_intent']


def test_fd_code_never_calls_gradient_or_other_AD_and_fixed_resource_caps():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    forbidden={'gradient','grad','backward','jacobian','jacrev','jacfwd','jvp','vjp','setattr','setprofile'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert m.contract()['signed_requests']==240 and m.contract()['pairs']==120
    assert m.contract()['max_new_compute']==m.src.caps(240,0)
    assert m.contract()['output_bytes']==128*1024**2


def test_reduction_roundoff_is_not_numpy_bitwise_gate_but_rejects_false_mean():
    y=np.full(192,np.float32(.25));g=np.zeros(192,np.float32);d=y-g;s=d*d
    q=np.asarray([np.nextafter(s.mean(),np.float32(1))],np.float32)
    arrays={k:m.signal.blob(v) for k,v in dict(endpoint=y,goal=g,residual=d,square=s,loss=q).items()}
    out=m.check_objective(arrays,y,g)
    assert out['native_reduction_difference']!=0 and not out['bitwise_GPU_mean_replayed']
    arrays['loss']=m.signal.blob(np.asarray([1.],np.float32))
    with pytest.raises(ValueError):m.check_objective(arrays,y,g)
