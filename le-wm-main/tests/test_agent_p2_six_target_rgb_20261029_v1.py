"""RAM/store/PNG ABI fixtures only; no production strict, model or physics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_six_target_rgb_20261029_v1 as m


def prepare_fixture(missing=False):
    rows=[]
    for i in range(24):
        slot,j=divmod(i,4);root=slot//3;cp=m.roster.CP[j];present=not (missing and slot==0 and cp>5)
        state=[float(cp),2.,3.,4.,.1,0.,0.];frame=None if not present else dict(step=cp,png=f'r{i}.png',state=state,snapshot=dict(state=state),file_sha256='a'*64,pixel_sha256='b'*64)
        rows.append(dict(index=i,slot=slot,root=root,condition='off',arm=m.roster.ARMS[slot%3],cp_public_step=cp,
            status='available_unencoded' if present else 'missing_physical_checkpoint',reason=None if present else 'native_checkpoint_not_reached',
            frame=frame,response=None if not present or not cp else dict(step=cp,state=state,reward=-1.,terminated=False,truncated=False),
            native_goal=[0.,0.,0.,0.,0.,0.,0.],task_id=f'task{root}',selection_digest=f'seal{slot}',
            raw15_sha256='c'*64,raw_prefix_bits='00'*cp,full_raw_bits='00'*120,
            predicted192=np.full(192,.125*root+.0625*j,np.float32).tolist(),prediction_pointer=dict(array='actual-token-prediction',index=[0,0,j-1]),
            prediction_origin='observed_history' if not cp else 'paid_actual_deployment_tokens',
            goal192=np.full(192,root+.5,np.float32).tolist(),goal_source=dict(fixture=True),
            target192=np.full(192,.125*slot,np.float32).tolist(),task_sha256='d'*64,blind_sha256='e'*64,
            original_z_sha256='f'*64,original_a_sha256='1'*64,pixel_source=None if not present else dict(frame,path=f'/fixture/r{i}.png'),
            source_time=None if slot%3!=1 else dict(source_raw_step=5,controller_terminal_equivalence=False),target_origin=dict(fixture=True),
            pixels=None,preprocessed=None,modules=[],latent=None,metrics=None,scientific_admission=False))
    n=sum(x['pixel_source'] is not None for x in rows)
    return dict(schema=m.NAME+'.prepare',rows=rows,contract=m.contract(),expected_model_hash='model',bindings={},
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS},
        imported_native_receipt=dict(counts={'fixture':150}),imported_paid_receipt={'fixture_paid':True},imported_controller_receipt={'fixture_controller':True})


def execute_fixture(p,tmp_path,monkeypatch):
    pix=np.zeros((224,224,3),np.uint8)
    monkeypatch.setattr(m.bridge,'load_pixels',lambda source:pix.copy())
    store=m.Store(tmp_path/(m.NAME+'_fixture'),m.Budget())
    store.put('binding.json',dict(prepared='/fixture/prepare.json',prepare_sha256='fixture'))
    store.put('initial_roster.json',p['rows'])
    rows=deepcopy(p['rows']);entries=[]
    def encode(pixels,row,scope):
        # Explicit algebraic fixture module IO, no neural forward.
        pre=m.bridge.pixels.preprocess_pixels(pixels);row['preprocessed']=store.array(f'r{row["index"]}_pre',pre)
        z=np.asarray(row['predicted192'],np.float32)+(np.float32(.125) if row['cp_public_step'] else np.float32(0))
        hidden=np.repeat(z[None,None],257,axis=1);projected=z[None]
        mods=[]
        for kind,inp,out in [('encoder',pre.reshape(1,3,224,224),hidden),('projector',hidden[:,0],projected)]:
            e=dict(index=len(entries),scope=deepcopy(scope),resource=kind,completed=False)
            store.event(dict(kind='module_attempt',entry=deepcopy(e)))
            e['inputs']=[store.array(f'm{e["index"]}_in',inp)];e['output']=store.array(f'm{e["index"]}_out',out);e['completed']=True
            store.event(dict(kind='module_return',entry=deepcopy(e)));entries.append(e);mods.append(e)
        row['modules']=deepcopy(mods);return z
    m.process(p,rows,store,encode)
    events=[m.read(store.path/f'event_{i:04d}.json') for i in range(store.index)]
    r=dict(schema=m.NAME+'.report',contract=m.contract(),completed=True,errors=[],model_before='model',model_after='model',
        gpu_peaks=dict(allocated=1,reserved=1,guard_checks=1),host=dict(device='cuda:0'),rows=rows,modules=entries,counts=m.bridge.counts(entries),resources=m.accounting(p,rows,entries))
    return r,events,lambda d:m.bridge.rgb.load_array(d,lambda n:(store.path/n).read_bytes()),store


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    patch=pytest.MonkeyPatch();p=prepare_fixture();tmp=tmp_path_factory.mktemp('six_rgb')
    r,e,load,store=execute_fixture(p,tmp,patch)
    yield p,r,e,load,store
    patch.undo()


def test_protocol_exact_no_production_scope():
    p=m.contract();assert p['rows']==p['pairs']==24 and p['max_encoder_API']==24
    assert p['new_predict']==p['new_AD']==p['new_physics']==p['new_goal_encode']==0
    assert p['allocated_cap']==3*1024**3 and p['reserved_cap']==4*1024**3
    assert not any(p[k] for k in ('scientific_admission','P1_admission','P2_mechanism','equal_FLOPs'))


def test_full24_three_layer_join(saved):
    p,r,e,load,_=saved;before=deepcopy(p);out=m.validate(p,r,e,load)
    assert p==before and len(out['rows'])==len(out['pairs'])==24 and out['winner'] is None
    assert out['denominator']==dict(slots=6,rows=24,pairs=24)
    for row,spec,actual in zip(out['rows'],p['rows'],r['rows'],strict=True):
        z=load(actual['latent']);pred=np.asarray(spec['predicted192'],np.float32);goal=np.asarray(spec['goal192'],np.float32)
        assert row['model_task_Q']==m.comparison.mse64(pred,goal)
        assert row['actual_task_Q']==m.comparison.mse64(z,goal)
        assert row['prediction_vs_encoded_MSE']==m.comparison.mse64(pred,z)
        assert row['native']['state7_order'][-2:]==['agent_vx','agent_vy']
        assert row['actual_task_reduction_from_cp0']==out['rows'][4*row['slot']]['actual_task_Q']-row['actual_task_Q']
        assert not row['physical_benefit_claim'] and row['cp_public_step']==spec['cp_public_step']
    assert all(x['comparable'] and x['cross_target_tracking_loss_difference'] is None for x in out['pairs'])


def test_missing_no_encoding_no_padding(tmp_path,monkeypatch):
    p=prepare_fixture(missing=True);r,e,load,_=execute_fixture(p,tmp_path,monkeypatch);out=m.validate(p,r,e,load)
    assert len(out['rows'])==24 and r['counts']['encoder']==dict(attempted=22,completed=22)
    for i in (2,3):assert out['rows'][i]['missing'] and out['rows'][i]['actual_task_Q'] is out['rows'][i]['native'] is None
    assert sum(x['comparable'] for x in out['pairs'])==20


@pytest.mark.parametrize('fault',['schema','partial','error','contract','model','gpu','slots','roster','status','modules','scope','count','unknown_event','event_order','global_modules','goal'])
def test_saved_mutations(saved,fault):
    p,r,e,load,_=saved;p,r,e=deepcopy(p),deepcopy(r),deepcopy(e)
    if fault=='schema':r['schema']='old.report'
    elif fault=='partial':r['completed']=False
    elif fault=='error':r['errors']=['error']
    elif fault=='contract':r['contract']['rows']=12
    elif fault=='model':r['model_after']='changed'
    elif fault=='gpu':r['gpu_peaks']['reserved']=m.RESERVED+1
    elif fault=='slots':r['rows'].pop()
    elif fault=='roster':r['rows'][1]['raw15_sha256']='changed'
    elif fault=='status':r['rows'][0]['status']='not_started'
    elif fault=='modules':r['rows'][0]['modules'].pop()
    elif fault=='scope':r['rows'][0]['modules'][0]['scope']['step']=5
    elif fault=='count':r['counts']['predict_API']['attempted']=1
    elif fault=='unknown_event':e.append({'kind':'extra'})
    elif fault=='event_order':e[0],e[1]=e[1],e[0]
    elif fault=='global_modules':r['modules'].pop()
    else:
        p['rows'][4]['goal192'][0]+=1;r['rows'][4]['goal192']=p['rows'][4]['goal192']
    with pytest.raises(ValueError):m.validate(p,r,e,load)


@pytest.mark.parametrize('fault',['cp0','preprocess','CLS','latent'])
def test_array_mutations(saved,fault):
    p,r,e,load,_=saved
    target=(r['rows'][0]['latent'] if fault in ('cp0','latent') else
        r['rows'][0]['preprocessed'] if fault=='preprocess' else r['rows'][0]['modules'][1]['inputs'][0])
    def bad(d):
        x=load(d)
        if d==target:x=x.copy();x.flat[0]+=np.float32(1)
        return x
    with pytest.raises(ValueError):m.validate(p,r,e,bad)


def test_run_refuses_before_verify(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized verify'))
    with pytest.raises(ValueError,match='MAIN authorization'):m.run(tmp_path/'m','h',tmp_path/'out')


def test_metadata_exclusive_roundtrip(tmp_path,monkeypatch):
    p=prepare_fixture();path=tmp_path/(m.NAME+'_prepare.json');ref=m.write_metadata(path,p)
    monkeypatch.setattr(m,'build',lambda **kw:deepcopy(p));p['arguments']={}
    # Use a separate source-shaped dict for prepare/verify control-flow test.
    path2=tmp_path/(m.NAME+'_verify.json');ref2=m.write_metadata(path2,p)
    assert m.verify(path2,ref2['sha256'])==p
    with pytest.raises(FileExistsError):m.write_metadata(path,p)
    assert m.read(path,ref['sha256'])['schema']==m.NAME+'.prepare'
    with pytest.raises(ValueError):m.write_metadata(tmp_path/'wrong.json',p)


def test_budget_wall_reserve_floor():
    now=[0.];b=m.Budget(clock=lambda:now[0],free=lambda:10*m.CAP);b.used=m.CAP-m.FINAL
    with pytest.raises(ValueError):b.check(1)
    b.check(1,final=True);b.used=0;now[0]=m.WALL+1
    with pytest.raises(ValueError):b.check()
    b.check(final=True);b.free=lambda:0
    with pytest.raises(ValueError):b.check(final=True)


def test_cli_fixture_dispatch(monkeypatch,capsys,tmp_path):
    calls=[];monkeypatch.setattr(m,'build',lambda *args:calls.append(args) or dict(fixture=True))
    monkeypatch.setattr(m,'write_metadata',lambda p,v:dict(fixture=v))
    m.main(['prepare','--native-output',str(tmp_path),'--native-report-sha','rh','--native-receipt',str(tmp_path/'c'),'--native-receipt-sha','ch','--output',str(tmp_path/'p')])
    assert len(calls)==1 and json.loads(capsys.readouterr().out)==dict(fixture=dict(fixture=True))


def test_real_metadata_abi_only_no_values_or_consumer():
    root=m.ROOT/'results';nr=m.read(root/'agent_p2_six_target_physical_20261029_v1_MAIN01/report.json',m.NATIVE_REPORT_SHA)
    nc=m.read(root/'agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json',m.NATIVE_RECEIPT_SHA)
    pr=m.read(root/'agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/report.json',m.PAID_REPORT_SHA)
    assert nr['schema']==m.physical.NAME+'.report' and nc['fixed_denominator']==6 and len(pr['rows'])==6
    assert [[c['step'] for c in row['checkpoints']] for row in nc['rows']]==[list(m.roster.CP)]*6
    assert m.sha(m.physical.__file__)==m.PHYSICAL_SOURCE_SHA


def test_cold_import_no_neural():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-B','-c',code],check=True)


@pytest.mark.parametrize('fault',[None,'extra','receipt','wall','inventory','order','late_input','resource'])
def test_file_level_consume_fixture(tmp_path,monkeypatch,fault):
    p=prepare_fixture();r,e,load,store=execute_fixture(p,tmp_path,monkeypatch)
    seal={k:r[k] for k in ('rows','modules','counts','errors')}
    h=store.put('native_seal.json',seal)
    r.update(prepared='/fixture/prepare.json',prepare_sha256='fixture',native_seal_sha256=h,
        artifact_sha256=deepcopy(store.hashes),artifact_order=list(store.order),event_count=store.index,wall_before_report=1)
    if fault=='inventory':r['artifact_sha256']['unknown']='0'*64
    elif fault=='order':r['artifact_order'][2],r['artifact_order'][3]=r['artifact_order'][3],r['artifact_order'][2]
    elif fault=='late_input':r['artifact_order'][0],r['artifact_order'][-1]=r['artifact_order'][-1],r['artifact_order'][0]
    elif fault=='resource':r['resources']['encoder_samples']=23
    store.put('report.json',r,final=True);rh=m.sha(store.path/'report.json')
    done=dict(schema=m.NAME+'.completion',report_sha256=rh,complete=True,wall_inclusive=2,boundary=m.BOUNDARY)
    if fault=='receipt':done['report_sha256']='wrong'
    elif fault=='wall':done['wall_inclusive']=m.WALL+1
    store.put('completion.json',done,final=True)
    if fault=='extra':store.put('unexpected.json',{})
    monkeypatch.setattr(m,'verify',lambda *a:p)
    if fault is not None:
        with pytest.raises(ValueError):m.consume(store.path,rh)
    else:
        out=m.consume(store.path,rh)
        assert out['denominator']==dict(slots=6,rows=24,pairs=24)
        assert out['resources']['encoder_samples']==out['resources']['projector_samples']==24
        assert out['original_controller_receipt']==p['imported_controller_receipt']
        assert out['inherited_paid_receipt']==p['imported_paid_receipt']
