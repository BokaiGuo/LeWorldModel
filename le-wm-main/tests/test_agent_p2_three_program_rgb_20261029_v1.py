"""RAM tensor/callback doubles and saved arithmetic only; no torch or environment execution."""
from copy import deepcopy
import hashlib
from pathlib import Path
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_three_program_rgb_20261029_v1 as m
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as enc
import test_agent_p2_three_program_physical_20261029_v1 as phys

def fixture(missing=()):
    p,_,modelprep,_=phys.fixture()
    actual=[]
    for i,plan in enumerate(p['plans']):
        mp=plan['planner_plan'];ctx=deepcopy(mp['root_context']);edges=[]
        for j in range(3):
            out=deepcopy(ctx);out['latent_history'][-1]=np.full(192,(j+1)*.125,np.float32).tolist()
            e=dict(index=j,input_context=ctx,output_context=out,commands=mp['commands'][j*5:j*5+5],
                input_context_sha256=m.planner.digest(ctx),goal_sha256=m.planner.digest(mp['goal']),
                task_id=mp['task_id'],source_report_sha256=mp['source_report_sha256'])
            e['witness_sha256']=m.planner.digest(e);edges.append(e);ctx=out
        mp['edges']=edges;mp['predicted_terminal']=ctx['latent_history'][-1]
        mp['predicted_Q']=m.bridge.mse(mp['predicted_terminal'],mp['goal'])
        mp['selection_digest']=m.planner.digest({k:v for k,v in mp.items() if k not in ('selection_digest','planner_report_sha256','prepare_sha256')})
        plan['selection_digest']=m.producer.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        # Missing is always a suffix of checkpoints: native stop is never padded.
        stop=min([s-1 for pi,s in missing if pi==i],default=15)
        row=dict(index=i,root_slot=plan['root_slot'],condition='off',arm=plan['arm'],task_id=plan['task_id'],
            plan=deepcopy(plan),selection_digest=plan['selection_digest'],status='horizon_completed' if stop==15 else 'native_stopped',
            own_returned=stop,checkpoints=[],trajectory=[],responses=[],missing_checkpoints=[])
        for step in (5,10,15):
            if step>stop:row['missing_checkpoints'].append(step);continue
            state=deepcopy(plan['native_goal']);state[0]+=step
            cp=dict(step=step,png=f'p{i}_s{step}.png',file_sha256=phys.h(('file',i,step)),
                pixel_sha256=phys.h(('pixels',i,step)),state=state,snapshot=dict(RAM=True))
            row['checkpoints'].append(cp);row['trajectory'].append(dict(cp,phase='plan',overlay_index=0,condition='off'))
            metric=m.producer.native.metrics(plan['native_goal'],state)
            row['responses'].append(dict(step=step,state=state,native_metrics=metric['native'],
                reward=metric['native7_reward'],native_success=metric['native_success'],terminated=metric['native_success'],truncated=False))
        actual.append(row)
    p['planner_authority']['plans_sha256']=m.producer.digest(p['plans'])
    strict=dict(schema=m.producer.NAME+'.consume',status=m.producer.RECORDED,fixed_denominator=4,known_roots=2,
        errors=[],P1_admission=False,P2_admission=False,physical_release=False,rows=actual)
    return p,strict,modelprep

def prepared(missing=()):
    rows=m.fixed_rows(*fixture(missing),'/RAM/physical')
    n=sum(x['source'] is not None for x in rows)
    return dict(rows=rows,contract=m.contract(),expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})

def execute(monkeypatch,missing=(),tokens=257,forbidden=None):
    p=prepared(missing);rows=deepcopy(p['rows']);s=enc.old.Memory();model=enc.Model(tokens);torch=enc.fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(3,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(enc.Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(enc.Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    assert model.predict==original and not errors and not guard.blocked
    r=dict(rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,modules=trace.entries,guards=guard.entries,
        counts=m.bridge.counts(trace.entries))
    return p,r,s

@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)

def test_fixed12_same_time_Q_join_and_full_actual_tensor_chronology(replay,monkeypatch):
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels);p,r,s=replay
    rows=m.validate(p,r,s.events,s.load)
    assert len(rows)==12 and len(r['modules'])==len(r['guards'])==24
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=12,completed=12)
    assert [x['checkpoint'] for x in rows]==[5,10,15]*4
    assert len(set(x['model_Q'] for x in rows[:3]))==3
    assert all(x['native_metrics'] is not None and not x['physical_benefit_claim'] for x in rows)

@pytest.mark.parametrize('missing',[((0,15),),((0,10),(3,5)),tuple((i,5) for i in range(4))])
def test_missing_fixed12_not_early_terminal_substitution(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing)
    result=m.validate(p,r,s.events,s.load)
    assert len(result)==12
    assert p['expected_calls']['encoder']==sum(not x['latent_missing'] for x in result)
    for row in result:
        if row['latent_missing']:
            assert row['actual_encoded_Q'] is row['prediction_vs_encoded_MSE'] is row['native_metrics'] is None
            assert row['model_Q'] is not None

@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_no_new_prediction_AD_or_action_encoder(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)

def test_hidden_shape_not_relaxed(monkeypatch):
    with pytest.raises(ValueError):execute(monkeypatch,tokens=197)

@pytest.mark.parametrize('fault',['drop','order','task','model','raw','edge','cpstep','state','response','missing','early','terminalQ'])
def test_plan_frame_action_time_model_mutations(fault):
    p,s,mp=fixture();a=s['rows'][0];plan=p['plans'][0]
    if fault=='drop':s['rows'].pop()
    elif fault=='order':s['rows'].reverse()
    elif fault=='task':a['task_id']='bad'
    elif fault=='model':mp['expected_model_hash']='bad'
    elif fault=='raw':plan['commands'][0][0]=.75
    elif fault=='edge':plan['planner_plan']['edges'][0]['commands'][0][0]=.75
    elif fault=='cpstep':a['checkpoints'][0]['step']=10
    elif fault=='state':a['checkpoints'][0]['state'][0]+=1
    elif fault=='response':a['responses'][0]['reward']+=1
    elif fault=='missing':a['missing_checkpoints']=[5]
    elif fault=='early':a['own_returned']=4
    else:plan['planner_plan']['predicted_Q']+=1
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(p,s,mp,'/RAM/o')

@pytest.mark.parametrize('fault',['event','drop_attempt','order','shape','preprocessed','pixel','latent','spec','guard','missing','count','blocked'])
def test_saved_full_array_source_receipt_chronology_mutations(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=enc.old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',enc.pixels);row=r['rows'][0]
    if fault=='event':events.insert(0,dict(kind='BAD_EVENT'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='order':events[2],events[3]=events[3],events[2]
    elif fault=='shape':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='preprocessed':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixel':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='latent':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='spec':row['model_Q']+=1
    elif fault=='guard':r['guards'].pop()
    elif fault=='missing':row['source']=None
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)

def test_unauthorized_run_before_io(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not read'))
    with pytest.raises(ValueError):m.run('/absent','a'*64,'/absent','bad')

def test_actual_saved_physical_schema_and_per_step_prediction_Q():
    # Saved JSON arithmetic, no prepare/run/physics/model. Real report and full receipt are immutable.
    root=m.ROOT/'results';rp=root/'agent_p2_three_program_physical_20261029_v1_MAIN01/report.json'
    cp=root/'agent_p2_three_program_physical_20261029_v1_consume_MAIN01.json'
    if not rp.exists() or not cp.exists():pytest.skip('MAIN actual saved assets not present')
    assert m.sha(rp)=='49b682d0ca326590c88ef69d74eecfdd9d5c19f21667e790a7c1458aac98e064'
    assert m.sha(cp)=='95c967c95c8ff00e0c2d4ab5c801afbab708095479c96986319dc103933208ac'
    r=m.read(rp);p=m.read(r['execution_binding']['prepared']);strict=m.read(cp)
    mr=m.read(Path(p['arguments']['planner_output'])/'report.json');mp=m.read(mr['prepared'])
    rows=m.fixed_rows(p,strict,mp,rp.parent)
    assert len(rows)==12 and all(x['source'] is not None for x in rows)
    for x in rows:
        m.bridge.load_pixels(x['source'])
        assert x['model_Q']==m.bridge.mse(x['edge']['output_context']['latent_history'][-1],x['goal192'])

def test_metadata_cap_and_exclusive_write(tmp_path):
    path=tmp_path/(m.NAME+'_unit.json')
    m.write(path,dict(status='RAM'))
    with pytest.raises(ValueError):m.write(path,{})
    assert m.META>=2*1024**2 and m.contract()['max_calls']['predict_API']==0


def actual_arguments():
    root=m.ROOT/'results'
    return dict(physical_output=str(root/'agent_p2_three_program_physical_20261029_v1_MAIN01'),
        physical_report_sha='49b682d0ca326590c88ef69d74eecfdd9d5c19f21667e790a7c1458aac98e064',
        physical_receipt=str(root/'agent_p2_three_program_physical_20261029_v1_consume_MAIN01.json'),
        physical_receipt_sha='95c967c95c8ff00e0c2d4ab5c801afbab708095479c96986319dc103933208ac',
        physical_source_sha='b0e7ae15d4810963a64916b1da16e34ae1017b189156a93b635d73520e2666e5')


def test_actual_full_saved_build_then_verify_no_runtime(tmp_path):
    args=actual_arguments()
    if not Path(args['physical_receipt']).exists():pytest.skip('MAIN saved assets absent')
    p=m.build(**args)
    assert p['expected_calls']['encoder']==12 and len(p['rows'])==12
    path=tmp_path/(m.NAME+'_actual_saved_unit.json');written=m.write(path,p)
    assert m.verify(path,written['sha256'])==p


@pytest.mark.parametrize('fault',['source','report','receipt','whole_receipt'])
def test_external_authority_failclosed_before_model(monkeypatch,tmp_path,fault):
    args=actual_arguments()
    if not Path(args['physical_receipt']).exists():pytest.skip('MAIN saved assets absent')
    if fault=='source':args['physical_source_sha']='0'*64
    elif fault=='report':args['physical_report_sha']='0'*64
    elif fault=='receipt':args['physical_receipt_sha']='0'*64
    else:
        real=m.read(args['physical_receipt']);fake=deepcopy(real);fake['counts']={}
        path=tmp_path/(m.NAME+'_bad_receipt.json');m.write(path,fake)
        args.update(physical_receipt=str(path),physical_receipt_sha=m.sha(path))
        monkeypatch.setattr(m.producer,'consume',lambda *a:real)
    with pytest.raises(ValueError):m.build(**args)

