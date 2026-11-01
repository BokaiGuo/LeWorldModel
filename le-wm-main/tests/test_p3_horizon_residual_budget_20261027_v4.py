"""Array/metadata fixtures only: no neural or simulator execution."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p3_horizon_residual_budget_20261027_v4 as m


def signals(a,b):return {'2':{'chi':a},'3':{'chi':b}}


@pytest.mark.parametrize('a,b,blocks',[(1,0,8),(1,1,3),(0,0,3),(.8,.4,5),(.2,.8,3)])
def test_dynamic(a,b,blocks):
    x=m.allocate('dynamic',signals(a,b))
    assert x['blocks']['3']==blocks
    assert 2*x['candidates']['2']+3*x['candidates']['3']==60


@pytest.mark.parametrize('a,b',[(1,0),(0,1),(.4,.7)])
def test_uniform(a,b):assert m.allocate('uniform',signals(a,b))['candidates']=={'2':15,'3':10}


@pytest.mark.parametrize('a,b',[(float('nan'),0),(-1,0),(0,2)])
def test_invalid_signal(a,b):
    with pytest.raises(ValueError):m.allocate('dynamic',signals(a,b))


@pytest.mark.parametrize('h',[2,3])
def test_residual_column_image_and_nominal(h):
    j=np.zeros((192,10*h));j[0,0]=1;q=np.eye(192);g=np.zeros(192);g[0]=2;g[1]=2
    x=m.residual_signal(j,np.zeros(192),g,q,h)
    assert x['chi']==pytest.approx(.5) and x['rank']==1
    n=np.zeros(192);n[1]=2
    assert m.residual_signal(j,n,g,q,h)['chi']==pytest.approx(0)
    assert m.residual_signal(j*0,g,g,q,h)['chi']==0
    assert m.residual_signal(j*0,np.zeros(192),g,q,h)['chi']==1


@pytest.mark.parametrize('h',[2,3])
def test_pool_prefix_and_domain(h):
    a=m.pool(0,h,3);b=m.pool(0,h,30)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert np.count_nonzero(a[0])==0
    assert not np.array_equal(m.pool(1,h,2)[1],a[1])
    for raw in b:
        assert raw.shape==(5*h,2) and np.max(np.abs(raw))<1
        assert np.linalg.norm(np.arctanh(raw))<=np.sqrt(5*h)*.75+1e-12


def fake_model():
    callback=lambda x:x
    return SimpleNamespace(predict=callback,action_encoder=SimpleNamespace(forward=callback),encoder=SimpleNamespace(forward=callback))


def tensor(kind='predict',device='cuda'):
    return SimpleNamespace(shape=(1,3,192 if kind=='predict' else 10),device=SimpleNamespace(type=device))


@pytest.mark.parametrize('kind',['predict','action_encoder'])
def test_shared_guard_nonrefundable(kind):
    model=fake_model();original=model.predict;log=[]
    with m.NeuralCap(model,log.append) as cap:
        with cap.scope('shared'):
            fn=model.predict if kind=='predict' else model.action_encoder.forward
            for _ in range(10):fn(tensor(kind))
            with pytest.raises(ValueError):fn(tensor(kind))
    assert cap.receipt()['attempted'][kind]==10 and cap.blocked==1
    assert model.predict is original


def test_joint_cap_and_scope():
    model=fake_model()
    with m.NeuralCap(model,lambda e:None) as cap:
        with pytest.raises(ValueError):model.predict(tensor())
        for role,count in [('shared',10),('uniform',66),('dynamic',66)]:
            with cap.scope(role):
                for _ in range(count):
                    model.action_encoder.forward(tensor('action_encoder'));model.predict(tensor())
                with pytest.raises(ValueError):model.predict(tensor())
        with pytest.raises(ValueError):model.encoder.forward(tensor())
    assert cap.receipt()['attempted']=={'predict':142,'action_encoder':142}


def test_failed_entry_is_durable_and_charged():
    model=fake_model();log=[]
    def fail(x):raise RuntimeError('fixture')
    model.predict=fail
    with m.NeuralCap(model,log.append) as cap:
        with cap.scope('uniform'):
            with pytest.raises(RuntimeError):model.predict(tensor())
            with pytest.raises(ValueError):model.predict(tensor(device='cpu'))
    assert cap.receipt()['attempted']['predict']==1
    assert cap.receipt()['completed']['predict']==0
    assert log[0]['completed'] is False and 'error' in log[-1]


def packet():
    return dict(root_context={'latent_history':np.zeros((3,192)).tolist(),'action_history':np.zeros((2,10)).tolist()},
                goal=np.zeros(192).tolist(),projection=np.eye(192).tolist())


@pytest.mark.parametrize('mode',['uniform','dynamic'])
def test_full_search_and_double_replay(mode):
    work=[];events=[]
    def replay(context,actions):
        work.append(len(actions)//5);return deepcopy(context)
    result=m.search(packet(),0,mode,signals(1,0),replay,events.append)
    assert sum(work)==64 # tie -> H2 zero, four verification predicts
    assert result['winner']['horizon']==2 and result['winner']['index']==0
    assert result['whole_and_segmented_exact']
    assert len(result['candidates'])==sum(result['allocation']['candidates'].values())
    assert all(e['completed'] for e in events if e.get('completed'))


def test_failure_no_winner_or_retry():
    events=[]
    def fail(c,a):raise RuntimeError('fixture')
    with pytest.raises(RuntimeError):m.search(packet(),0,'uniform',signals(1,0),fail,events.append)
    assert len(events)==1 and events[0]['completed'] is False


def test_cap_arithmetic():
    c=m.contract()
    assert c['shared_predict_cap']==2*2+2*3
    assert c['pair_predict_cap']==10+2*(60+6)
    assert c['four_root_neural_entry_cap']==4*2*142
    assert c['mathematical_vjp_directions_per_root']==2*192
    assert c['P1_admission'] is c['P3_admission'] is False


def test_real_prepare_verify_exclusive_and_tamper(tmp_path):
    # Actual saved source inputs; no neural imports, no model loading or SVD.
    from pathlib import Path
    root=Path(m.__file__).resolve().parents[2];out=tmp_path/'prepared'
    receipt=m.prepare(root,out);sha=m.io.sha((out/'prepare.json').read_bytes())
    body,identity=m.validate_prepared(out,sha)
    assert receipt['protocol_id']==identity and len(body['packets'])==4
    assert all(set(p)=={'root_sha256','root_context','goal','projection','input_sha256'} for p in body['packets'])
    assert receipt['model_loaded'] is False
    with pytest.raises(ValueError):m.prepare(root,out)
    (out/'protocol.json').write_text('{}')
    with pytest.raises(ValueError):m.validate_prepared(out,sha)


def exporter_fixture(tmp_path,monkeypatch):
    body=dict(roots=[str(i)*64 for i in range(4)],packets=[])
    rows=[];records=[]
    for i,root in enumerate(body['roots']):
        p=packet();p.update(root_sha256=root,input_sha256='fixture');body['packets'].append(p)
        compilation={}
        for h in (2,3):
            j=np.zeros((192,10*h));n=np.zeros(192)
            compilation[str(h)]=dict(jacobian=j.tolist(),nominal=n.tolist(),signal=m.residual_signal(j,n,p['goal'],p['projection'],h))
        sig={h:c['signal'] for h,c in compilation.items()}
        for mode in m.METHODS:
            s=m.search(p,i,mode,sig,lambda c,a:deepcopy(c),lambda e:None)
            rows.append(dict(root_sha256=root,input_sha256='fixture',mode=mode,status='completed',search=s))
        events=[dict(kind=k,role=role,completed=True,context_tokens=3) for k in ('predict','action_encoder')
                for role,n in [('shared',10),('uniform',64),('dynamic',64)] for _ in range(n)]
        for index,event in enumerate(events):event['index']=index
        logs=[]
        for event in events:logs.extend([dict(event,completed=False),event])
        logs.extend(dict(kind=kind,horizon=h,completed=True) for h in (2,3) for kind in ('jacobian','svd'))
        journal=tmp_path/f'{i}.jsonl';journal.write_text('\n'.join(m.io.canonical(e) for e in logs)+'\n')
        records.append(dict(root_sha256=root,compilation=compilation,accounting_exact=True,
            neural_receipt=dict(events=events,blocked=0),journal_path=str(journal),journal_sha256=m.io.sha(journal.read_bytes())))
    monkeypatch.setattr(m,'validate_prepared',lambda p,h:(body,'identity'))
    m.write(tmp_path/'execution_protocol.json',dict(prepared='fixture',prepare_sha256='fixture',protocol_id='identity'))
    report=dict(schema=m.NAME+'.report',status='cuda_completed',protocol_id='identity',rows=rows,roots=records,
        model_unchanged=True,failure=None,contract=m.contract(),P1_admission=False,P3_admission=False)
    return report


@pytest.mark.parametrize('damage',[None,'duplicate','actions','witness','budget','Q','chi'])
def test_eight_arm_source_bound_export(tmp_path,monkeypatch,damage):
    report=exporter_fixture(tmp_path,monkeypatch)
    s=report['rows'][0]['search']
    if damage=='duplicate':report['rows'].append(deepcopy(report['rows'][0]))
    if damage=='actions':s['candidates'][0]['actions'][0][0]=.5
    if damage=='witness':s['whole_replay_context']['latent_history'][0][0]=1
    if damage=='budget':report['roots'][0]['neural_receipt']['events'].pop()
    if damage=='Q':s['candidates'][0]['cost']=3
    if damage=='chi':report['roots'][0]['compilation']['2']['signal']['chi']=.4
    m.write(tmp_path/'report.json',report)
    if damage:
        with pytest.raises(ValueError):m.selected_actions(tmp_path)
    else:
        plans=m.selected_actions(tmp_path)
        assert len(plans)==8 and len({(p['root_sha256'],p['mode']) for p in plans})==8
        assert all(p['horizon']==2 and len(p['actions'])==10 and not p['P3_admission'] for p in plans)
