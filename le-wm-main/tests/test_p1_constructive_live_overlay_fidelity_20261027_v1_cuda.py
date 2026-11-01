"""Own-view chronology, label firewall and eight-call mock contracts; all RAM."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_live_overlay_fidelity_20261027_v1_cuda as m
from test_p1_constructive_live_overlay_fidelity_20261027_v1 import fixture


def latents(view=0):return [np.full(192,view+i,np.float32) for i in range(4)]
def commands():return np.linspace(-.8,.8,20,dtype=np.float32).reshape(10,2)


class Backend:
    def __init__(self):self.calls=[]
    def step(self,c,a):
        self.calls.append((deepcopy(c),a.copy()));out=deepcopy(c)
        out['latent_history']=c['latent_history'][1:]+[(np.asarray(c['latent_history'][-1],np.float32)+np.float32(.25)).astype(float).tolist()]
        out['action_history']=c['action_history'][1:]+m.base.d2p.normalized_history(a).astype(float).tolist()
        return out


def test_context_exact_three_observed_two_packets():
    z=latents();a=commands();c,outgoing,between=m.context(z,a)
    assert np.array_equal(c['latent_history'],z[:3])
    assert np.array_equal(between[:5],np.zeros((5,2))) and np.array_equal(between[5:],a[:5])
    assert np.array_equal(outgoing,a[5:])
    expected=m.base.d2p.normalized_history(between)
    assert np.array_equal(c['action_history'],expected) and np.any(expected[0]!=0)


def test_terminal_label_never_reaches_model():
    b=Backend();z=latents();r=m.diagnose(z,commands(),b,lambda e:None)
    c1,a1=deepcopy(b.calls[0]);z[3][:]=100
    r2=m.diagnose(z,commands(),b,lambda e:None)
    assert b.calls[1][0]==c1 and np.array_equal(b.calls[1][1],a1)
    assert r['factual_mse']!=r2['factual_mse'] and len(b.calls)==2


def test_factual_persistence_arithmetic_no_threshold():
    b=Backend();r=m.diagnose(latents(),commands(),b,lambda e:None)
    assert r['factual_mse']==.75**2 and r['persistence_mse']==1 and r['predicted_change_mse']==.25**2
    assert r['action_dependence_measured'] is False and r['counterfactual_action_control'] is None and r['P1_admission'] is False


def test_all8_predictions_no_dedup_and_own_view_labels():
    b=Backend();scores=[]
    for branch in range(4):
        for view in range(2):scores.append(m.diagnose(latents(view*10),commands(),b,lambda e:None)['factual_mse'])
    assert len(b.calls)==8 and scores==[.5625]*8
    assert b.calls[0][0]['latent_history'][0]!=b.calls[1][0]['latent_history'][0]


@pytest.mark.parametrize('fault',['len','shape','nan','nonfloat32','cap'])
def test_invalid_inputs_fail_before_neural(fault):
    z=latents();a=commands();b=Backend()
    if fault=='len':z.pop()
    elif fault=='shape':z[0]=np.zeros(191,np.float32)
    elif fault=='nan':z[1][0]=np.nan
    elif fault=='nonfloat32':z[0]=z[0].astype(float)
    else:a[0,0]=1.01
    with pytest.raises(ValueError):m.diagnose(z,a,b,lambda e:None)
    assert b.calls==[]


def paired_fixture():
    report,old,prep=fixture();branches=m.render.assemble(report,old,prep)
    rp=dict(branches=branches)
    r=dict(schema=m.render.NAME+'.report',status='live_overlay_pairs_sealed_not_admitted',errors=[],source_verified=True,
        contract=m.render.contract(),counts={k:dict(attempted=v,completed=v) for k,v in m.render.contract()['caps'].items()},rows=[])
    ev=[dict(kind='run_started')]
    for resource,n in m.render.contract()['caps'].items():
        for _ in range(n):ev.extend([dict(kind='operation_attempted',resource=resource),dict(kind='operation_completed',resource=resource)])
    for b,item in enumerate(branches):
        row=dict(branch=b,source_slot=item['source_slot'],arm=item['arm'],commands=item['commands'],status='replayed_and_paired',
            first_error=None,secondary_errors=[],off_exact=22,public_returned=20,pairs=[])
        r['rows'].append(row)
        ev.extend(dict(kind='off_receipt',branch=b,receipt=c) for c in item['receipts'])
        for phase,s in m.render.SELECTED:
            c=next(c for c in item['receipts'] if (c['phase'],c['step'])==(phase,s))
            pair=dict(branch=b,source_slot=item['source_slot'],arm=item['arm'],phase=phase,step=s,status='paired',
                views=dict(off=dict(file=c['png'],sha256=c['file_sha256'],pixel_sha256=c['pixel_sha256']),
                    on=dict(file=c['png']+'_on',sha256='onfile',pixel_sha256='onpixel')),
                proof=dict(before=dict(x=1),after=dict(x=1),snapshot_equal=True,goal_pose=m.render.DEFAULT_POSE,order=['off','on']))
            row['pairs'].append(pair);ev.append(dict(kind='pair_completed',receipt=pair))
    ev.append(dict(kind='run_finished',complete=True,counts=r['counts']));return r,ev,rp


def test_render_join_all32_fixed_roles():
    r,e,p=paired_fixture();images,branches=m.assemble(r,e,p)
    assert len(images)==32 and len(branches)==4
    assert [(i['view'],i['role']) for i in images[:8]]==[(v,role) for v in m.VIEWS for role in m.ROLES]
    assert [i['index'] for i in images]==list(range(32))


@pytest.mark.parametrize('fault',['pair_missing','mutated','wrong_default','off_missing','resource','view','commands','partial'])
def test_render_join_rejects(fault):
    r,e,p=paired_fixture()
    if fault=='pair_missing':r['rows'][0]['pairs'].pop()
    elif fault=='mutated':r['rows'][0]['pairs'][0]['proof']['after']['x']=2
    elif fault=='wrong_default':r['rows'][0]['pairs'][0]['proof']['goal_pose']=[0.,0.,0.]
    elif fault=='off_missing':e.pop(next(i for i,x in enumerate(e) if x['kind']=='off_receipt'))
    elif fault=='resource':e.pop(next(i for i,x in enumerate(e) if x['kind']=='operation_completed'))
    elif fault=='view':r['rows'][0]['pairs'][0]['views'].pop('on')
    elif fault=='commands':r['rows'][0]['commands']=[[1.,1.]]*10
    else:r['status']='partial_or_failed'
    with pytest.raises((ValueError,KeyError)):m.assemble(r,e,p)


def test_budget8_not24_no_hidden_controls():
    c=m.contract();assert c['encoder']==c['projector']==32
    assert c['predict']==c['action_encoder']==8 and c['action_encoder_packet_tokens']==24
    assert c['AD']==c['physics']==0 and c['selection'] is False


def test_ledger_predispatch8():
    ledger=m.PredictLedger(lambda e:None,lambda:None)
    for _ in range(8):ledger.end(ledger.begin('predict'))
    with pytest.raises(ValueError,match='fixed8'):ledger.begin('predict')


class Hook:
    def __init__(self,items,fn):self.items,self.fn=items,fn
    def remove(self):self.items.remove(self.fn)


class MockProjection:
    def __init__(self):self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn):self.pre.append(fn);return Hook(self.pre,fn)
    def register_forward_hook(self,fn):self.post.append(fn);return Hook(self.post,fn)
    def __call__(self,x):
        for f in self.pre:f(self,(x,))
        for f in self.post:f(self,(x,),x)
        return x


@pytest.mark.parametrize('shape,valid',[((3,192),True),((1,3,192),False),((1,192),False)])
def test_actual_flattened_pred_proj_receipts_without_neural(shape,valid):
    projection=MockProjection();model=SimpleNamespace(pred_proj=projection,predict=lambda x:projection(x))
    original=model.predict;x=SimpleNamespace(shape=shape,device=SimpleNamespace(type='cuda'));events=[]
    if valid:
        with m.prediction_receipts(model,events.append,lambda:None) as receipts:
            for _ in range(8):model.predict(x)
            with pytest.raises(ValueError,match='8 cap'):model.predict(x)
        assert len(receipts)==16 and all(r['completed'] for r in receipts)
        assert sum(r.get('tokens',0) for r in receipts)==24
    else:
        with pytest.raises(ValueError,match='flattened'):
            with m.prediction_receipts(model,events.append,lambda:None):model.predict(x)
    assert model.predict is original and not projection.pre and not projection.post
