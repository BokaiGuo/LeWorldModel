"""NumPy arithmetic / callback fixtures only; no neural model or physics."""
from copy import deepcopy
import inspect
import numpy as np
import pytest
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v1 as m
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as parent

inputs=parent.inputs
complete=parent.complete


class Arithmetic:
    def __init__(self,task,flat=False,fail=None):
        self.task=task;self.calls=[];self.fail=fail;self.j=np.zeros((192,10),np.float32)
        if not flat:self.j[:10]=np.diag(np.linspace(.2,1.1,10,dtype=np.float32))
    def after(self,c,a):
        a=np.asarray(a,np.float32);z=np.asarray(c['latent_history'][-1],np.float32)+self.j@a.reshape(10)
        token=((a-np.asarray(self.task['action_mean'],np.float32))/np.asarray(self.task['action_std'],np.float32)).reshape(10)
        return dict(c,latent_history=deepcopy(c['latent_history'][1:])+[z.tolist()],action_history=[c['action_history'][-1],token.tolist()])
    def step(self,c,a,label):
        self.calls.append(dict(kind='score',label=label,context=deepcopy(c),raw=np.asarray(a).copy()))
        if label==self.fail:raise RuntimeError('RAM score failure')
        return self.after(c,a)
    def gradient(self,c,a,label):
        self.calls.append(dict(kind='ADforward',label=label,context=deepcopy(c),raw=np.asarray(a).copy()))
        after=self.after(c,a);res=np.asarray(after['latent_history'][-1],np.float32)-np.asarray(self.task['goal'],np.float32)
        return dict(context=after,native_float32_loss=float(np.mean(res**2)),
            raw_gradient=((2*self.j.T@res)/np.float32(192)).reshape(5,2).astype(np.float32).tolist())


def make_request(complete,slot=0,overcast=False):
    s,c,r,_,_=deepcopy(complete);task=s['task']
    if overcast:
        rng=np.random.default_rng(17)
        for _ in range(1000):
            x=rng.normal(size=10);x*=np.nextafter(m.frozen.CAP,0)/np.linalg.norm(x)
            if np.linalg.norm(x)<=m.frozen.CAP<np.linalg.norm(x.astype(np.float32).astype(float)):break
        c['logits']=x.tolist();c['incumbent_raw']=m.frozen.chart(x).tolist()
        c['incumbent_domain']=m.frozen.incumbent_domain(dict(suffix_params=x.tolist(),suffix_commands=c['incumbent_raw'],
            suffix_solve=dict(params=x.tolist(),radius=m.frozen.CAP,norm=float(np.linalg.norm(x)))))
        # Reconstruct full parent numeric recurrence with zero derivatives and
        # flat costs to select ORIGINAL legal iterate0, not repaired float32.
        its=[];grads=[];cur=x.copy();ctx=deepcopy(r['selected_context'])
        for i in range(9):
            a=m.frozen.chart(cur);its.append(dict(iteration=i,logits=cur.tolist(),commands=a.tolist(),raw_bits=m.frozen.bits(a),context=ctx,Q=1.))
            if i<8:
                grads.append(dict(raw_gradient=np.zeros((5,2),np.float32).tolist(),chain=(1-a.astype(float)**2).tolist(),logit_gradient=[0.]*10))
                cur=m.frozen.proposal(cur,np.zeros(10))
        r.update(candidate_sha256=m.digest(c),iterates=its,gradients=grads,winner_iteration=0,commands=c['prefix']+its[0]['commands'],selected_Q=1.,selected_context=ctx)
    inc=m.selected_incumbent(c,r)
    root=np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)
    actual=np.concatenate((root,np.asarray(inc['prefix'],np.float32)))[-10:]
    observed=Arithmetic(task).after(task['root_context'],inc['prefix'])
    # All three actual cp5 image encodings may differ from imagined context.
    z=np.asarray(observed['latent_history'],np.float32);z+=np.float32(.04)
    observed['latent_history']=z.tolist()
    q=m.request(slot,task,observed,root.tolist(),actual.tolist(),inc,c,r)
    return q,c,r


@pytest.fixture
def req(complete):return make_request(complete)[0]


def execute(q,**kw):
    b=Arithmetic(q['task'],**kw);events=[];result=m.optimize(q,b,events.append)
    return result,events,b


def test_observed_context_is_used_all9_scores_all8_AD_not_parent_prediction(req):
    result,events,b=execute(req);assert m.verify_numeric(req,result,events)['numerical_verified']
    assert len(b.calls)==17 and sum(c['kind']=='score' for c in b.calls)==9
    assert sum(c['kind']=='ADforward' for c in b.calls)==8
    assert all(c['context']==req['observed_context'] for c in b.calls)
    assert result['commands'][:5]==req['incumbent']['prefix'] and len(result['commands'])==10
    assert result['new_suffix_Q']<=result['observed_incumbent_Q']
    assert result['actual_native_resources'] is result['physical_metrics'] is None
    assert result['parent_prediction_used_as_new_prediction'] is result['production_authority_verified'] is False
    assert np.array_equal(b.calls[0]['raw'],req['incumbent']['commands'])
    assert [c['label'] for c in b.calls]==['observed_iterate_0']+[v for i in range(1,9) for v in (f'observed_gradient_{i}',f'observed_iterate_{i}')]


def test_original_legal_overcast_iterate0_exactly_preserved(complete):
    q,_,_=make_request(complete,overcast=True);d=q['incumbent']['original_ridge_domain']
    assert d['solver_float64_norm']<=m.frozen.CAP<d['postcast_float64_norm']
    r,events,b=execute(q,flat=True)
    m.verify_numeric(q,r,events)
    assert r['winner_iteration']==0
    assert r['iterates'][0]['logits']==q['incumbent']['logits']
    assert r['iterates'][0]['raw_bits']==q['incumbent']['raw_bits']
    assert m.frozen.bits(b.calls[0]['raw'])==q['incumbent']['raw_bits']
    assert all(np.linalg.norm(v['logits'])<=m.frozen.CAP for v in r['iterates'][1:])


@pytest.mark.parametrize('bad',['state','reward','success','latent_imagined','history','raw','goal','projection','normalizer','prefix','incumbent','claim'])
def test_request_firewall_and_exact_history(req,bad):
    q=deepcopy(req)
    if bad in ('state','reward','success'):q[bad]=1
    elif bad=='latent_imagined':q['observed_context']['latent_history'][0][0]=float('nan')
    elif bad=='history':q['observed_context']['action_history'][0][0]+=.1
    elif bad=='raw':q['observed_raw10'][0][0]=.5
    elif bad=='goal':q['task']['goal'][0]+=.1
    elif bad=='projection':q['task']['projection'][0][0]=2
    elif bad=='normalizer':q['task']['action_std'][0]=2
    elif bad=='prefix':q['incumbent']['prefix'][0][0]=.75
    elif bad=='incumbent':q['incumbent']['commands'][0][0]=.75
    else:q['production_authority_verified']=True
    with pytest.raises(ValueError):m.validate_request(q)


@pytest.mark.parametrize('bad',['dropiterate','dropgradient','Q','winner','rawbits','gradient','chain','prefix','goal','nativecost','actualreceipt','chronology','unknown_event'])
def test_numerical_strict_rejects_mutation(req,bad):
    r,es,_=execute(req)
    if bad=='dropiterate':r['iterates'].pop()
    elif bad=='dropgradient':r['gradients'].pop()
    elif bad=='Q':r['iterates'][1]['Q']+=1
    elif bad=='winner':r['winner_iteration']=99
    elif bad=='rawbits':r['iterates'][1]['raw_bits']='00'*40
    elif bad=='gradient':r['gradients'][0]['raw_gradient'][0][0]+=.1
    elif bad=='chain':r['gradients'][0]['chain'][0][0]=2
    elif bad=='prefix':r['commands'][0][0]=.75
    elif bad=='goal':r['goal_sha256']='0'*64
    elif bad=='nativecost':r['scheduled_cost']['predict']=0
    elif bad=='actualreceipt':r['actual_native_resources']={}
    elif bad=='chronology':es[1],es[2]=es[2],es[1]
    else:es.insert(1,dict(kind='physical_success'))
    with pytest.raises(ValueError):m.verify_numeric(req,r,es)


def test_all4_panel_complete_and_failure_denominator(complete):
    qs=[make_request(complete,i)[0] for i in range(4)];backends=[]
    def factory(q):
        b=Arithmetic(q['task']);backends.append(b);return b
    r=m.panel(qs,factory,lambda e:None)
    assert r['complete'] and len(r['rows'])==4 and len(backends)==4
    assert sum(len(b.calls) for b in backends)==68
    assert m.contract()['total']['scalar_reverse']==32
    assert not r['production_ready'] and not r['P1_admission']
    partial=m.panel(qs,lambda q:Arithmetic(q['task'],fail='observed_iterate_2') if q['slot']==1 else Arithmetic(q['task']),lambda e:None)
    assert not partial['complete'] and len(partial['rows'])==4
    assert [x['status'] for x in partial['rows']]==['numeric_complete','failed_partial','not_started','not_started']
    assert len(partial['rows'][1]['events'])>0 and partial['rows'][1]['error'] is not None
    with pytest.raises(ValueError):m.panel(qs[:3],factory,lambda e:None)
    with pytest.raises(ValueError):m.panel([qs[0]]*4,factory,lambda e:None)


def test_parent_numeric_witness_cannot_be_replaced_by_old_Q_only(complete):
    _,c,r,_,_=deepcopy(complete)
    r['iterates'].pop()
    with pytest.raises(ValueError):m.selected_incumbent(c,r)


def test_module_explicitly_does_not_guess_external_ABI_or_load_model():
    c=m.contract()
    assert not c['production_ABI_bound'] and not c['production_run_authorized']
    assert c['inherited_compute'] is None and c['total_FLOPs'] is None
    text=inspect.getsource(m)
    assert 'load_compatible_pretrained' not in text and 'setattr(' not in text and 'torch.autograd' not in text
    assert not hasattr(m,'run') and not hasattr(m,'prepare')
