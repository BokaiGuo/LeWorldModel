"""RAM authority/hook/token lifecycle tests. No native renderer or neural model."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_target_raster_encoder_20261028_v1 as m


def npy(path,a):
    with path.open('xb') as f: np.save(f,a,allow_pickle=False)
    return dict(path=str(path),sha256=m.sha(path),bytes=path.stat().st_size)


@pytest.fixture
def static_seam(tmp_path,monkeypatch):
    """RAM layout reproduces actual v3 ABI, NOT a substituted real strict receipt."""
    original_renderer=m.renderer
    p=m.read_json(m.ROOT/'results/agent_p5_renderer_factorial_20261028_v3_prepare_MAIN01.json')
    # Source/native model inputs are real saved assets; only new static report and
    # strict invocation are RAM stubs. Never write these under results/MAIN.
    report_dir=tmp_path/'static'; report_dir.mkdir()
    rows=[]
    for state in p['native_manifest']['states']:
        s=state['state_id']; expected=original_renderer.native.view_state(state['source_state'])
        shots=[]
        for target in m.RASTERS:
            prefix=s+'_'+target
            rgb=npy(report_dir/(prefix+'.npy'),np.zeros((224,224,3),np.uint8))
            files={'rgb':dict(rgb,path=Path(rgb['path']).name)}
            for key in ('before','after'):
                path=report_dir/(prefix+'_'+key+'.json'); m.write_new(path,expected)
                files[key]=dict(path=path.name,sha256=m.sha(path),bytes=path.stat().st_size)
            shots.append(dict(target=target,files=files))
        rows.append(dict(state_id=s,shots=shots,restore=dict(readonly_difference=dict(path='space.current_time_step',source=.01,actual=0.))))
    manifest=tmp_path/'static_prepare.json'; report=report_dir/'report.json'; receipt=tmp_path/'static_receipt.json'
    m.write_new(manifest,p); m.write_new(report,dict(rows=rows))
    c=dict(schema=original_renderer.NAME+'.consume',status='all10_saved_static_target_views_verified',
        manifest_sha256=m.sha(manifest),report_sha256=m.sha(report),RAM_only=True,
        native_consumer=dict(denominator=10,baselines=5,Space_step=0,complete_dynamic_restore=False))
    m.write_new(receipt,c)
    m.write_new(Path(str(report_dir)+'.launch.json'),{'RAM_only':True})
    calls=[]
    def verify(path,h):
        assert isinstance(path,Path) and m.sha(path)==h
        calls.append('verify'); return p
    def consume(path,h,rpath,rh):
        assert isinstance(path,Path) and isinstance(rpath,Path) and m.sha(path)==h and m.sha(rpath)==rh
        calls.append('consume'); return deepcopy(c)
    stub=SimpleNamespace(**vars(original_renderer)); stub.verify=verify; stub.consume=consume
    monkeypatch.setattr(m,'renderer',stub)  # New module binding only; frozen module globals unchanged.
    own=SimpleNamespace(**vars(m.owner)); own.verify=lambda value:None
    monkeypatch.setattr(m,'owner',own)
    args=dict(static_manifest=manifest,static_manifest_sha256=m.sha(manifest),static_report=report,
        static_report_sha256=m.sha(report),static_receipt=receipt,static_receipt_sha256=m.sha(receipt))
    return args,c,calls,original_renderer


def test_prepare_named_actual_authorities_ten_inputs_coverage_join(static_seam):
    args,_,calls,original=static_seam
    before=(original.verify,original.consume,original.native.execute,original.native.protocol)
    value=m.prepare(**args)
    assert calls==['verify','consume']
    assert [x['image_key'] for x in value['inputs']]==list(m.KEYS)
    assert len(value['legacy_native_coverage'])==4
    assert value['protocol']['new_API']['official_encoder']==10
    assert value['scientific_admission'] is False and 'selected_word' not in value
    assert all(m.sha(path)==h for path,h in value['bindings'].items())
    m.verify(value)
    assert before==(original.verify,original.consume,original.native.execute,original.native.protocol)


def test_prepare_cli_six_named_fields_roundtrip(static_seam,tmp_path):
    args,_,_,_=static_seam; out=tmp_path/'own.json'
    command=['prepare','--manifest',str(out)]
    for k,v in args.items(): command.extend(['--'+k.replace('_','-'),str(v)])
    assert m.main(command)==0
    assert m.main(['verify','--manifest',str(out),'--manifest-sha256',m.sha(out)])==0


@pytest.mark.parametrize('fault',['receipt','status','denominator','step','bool_type','missing'])
def test_prepare_refuses_unstrict_or_mutated_receipt(static_seam,fault):
    args,c,_,_=static_seam
    if fault=='receipt': c['extra']='different consumer return'
    elif fault=='status': c['status']='pending'
    elif fault=='denominator': c['native_consumer']['denominator']=9
    elif fault=='step': c['native_consumer']['Space_step']=1
    elif fault=='bool_type':
        path=args['static_receipt']; v=m.read_json(path); v['native_consumer']['complete_dynamic_restore']=0
        path.write_text(json.dumps(v)); args['static_receipt_sha256']=m.sha(path)
    else: args['static_receipt']=args['static_receipt'].parent/'absent.json'
    with pytest.raises(ValueError): m.prepare(**args)


@pytest.mark.parametrize('fault',['rgb','state','order','sha','source'])
def test_prepare_refuses_inputs_and_authority_mutations(static_seam,monkeypatch,fault):
    args,_,_,_=static_seam
    if fault=='sha': args['static_report_sha256']='future'
    elif fault=='source': monkeypatch.setattr(m,'RENDERER_SHA','0'*64)
    else:
        p=args['static_report']; report=m.read_json(p)
        if fault=='order': report['rows'].reverse()
        else:
            rec=report['rows'][0]['shots'][0]['files']['rgb' if fault=='rgb' else 'after']
            member=p.parent/rec['path']
            if fault=='rgb': member.write_bytes(b'not numpy')
            else:
                value=m.read_json(member); value['bodies']['agent']['position'][0]+=1; member.write_text(json.dumps(value))
            rec.update(sha256=m.sha(member),bytes=member.stat().st_size)
        p.write_text(json.dumps(report)); args['static_report_sha256']=m.sha(p)
        # Receipt/consumer edge must also match before exercising the deeper input gate.
        c=m.read_json(args['static_receipt']); c['report_sha256']=args['static_report_sha256']
        args['static_receipt'].write_text(json.dumps(c)); args['static_receipt_sha256']=m.sha(args['static_receipt'])
        m.renderer.consume=lambda *a:deepcopy(c)
    with pytest.raises((ValueError,KeyError)): m.prepare(**args)


def test_actual_missing_strict_fails_closed(tmp_path):
    p=m.ROOT/'results/agent_p5_renderer_factorial_20261028_v3_prepare_MAIN01.json'
    with pytest.raises(ValueError):
        m.prepare(p,m.sha(p),tmp_path/'absent_report.json','0'*64,tmp_path/'absent_receipt.json','0'*64)


class Module:
    """Record-only Python hook object, no neural operations."""
    def __init__(self): self.pre=[]; self.post=[]; self.removed=0
    def handle(self,bank,hook):
        def remove(): bank.remove(hook); self.removed+=1
        return SimpleNamespace(remove=remove)
    def register_forward_pre_hook(self,h): self.pre.append(h); return self.handle(self.pre,h)
    def register_forward_hook(self,h): self.post.append(h); return self.handle(self.post,h)
    def call(self,fail=False):
        args=(SimpleNamespace(is_cuda=True,shape=(1,3,224,224)),)
        for h in self.pre: h(self,args)
        if fail: raise RuntimeError('RAM failed after attempted forward')
        for h in self.post: h(self,args,None)


@pytest.fixture
def scene(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); monkeypatch.setattr(m,'SHAPE',(2,3)); monkeypatch.setattr(m,'FLOOR',0)
    monkeypatch.setattr(m,'verify',lambda value:None); monkeypatch.delenv('PYTHONPATH',raising=False)
    (tmp_path/'results').mkdir(); source=tmp_path/'inert.py'; source.write_text('# RAM inert source\n')
    receipts={k:dict(RAM_only=True) for k in m.EXPECTED}
    value=dict(model_spec=dict(loader_manifest=dict(seals={str(source):m.sha(source)},expected_model_receipts=receipts)),
        legacy_native_coverage=[dict(word=w,coverage=.2,source='RAM metadata not physics') for w in m.WORDS],inputs=[])
    for k in m.KEYS:
        value['inputs'].append(dict(image_key=k,rgb=npy(tmp_path/(k+'.npy'),np.zeros((224,224,3),np.uint8))))
    return value,source,receipts


def setup(scene,monkeypatch,fault=None):
    value,source,receipts=scene; models={k:Module() for k in m.EXPECTED}
    def load(value):
        return models,{k:{'exact_match':True} for k in ('predictor','proprio_encoder')}, {'RAM':str(source)},receipts,
    def runtime(value):
        data=load(value)
        return (*data,lambda:{} if fault=='model' else receipts,lambda:{'RAM':str(source)})
    monkeypatch.setattr(m,'load_runtime',runtime)
    calls=[]
    def encode(model,rgb):
        calls.append(len(calls))
        if fault=='forbidden': models['predictor'].call()
        model.call(fail=fault=='forward')
        if fault=='replay': model.call()
        return np.full(m.SHAPE,len(calls),np.float32)
    monkeypatch.setattr(m,'encode',encode)
    return models,calls


def test_actual_run_control_flow_no_dedup_no_old_global_mutation(scene,tmp_path,monkeypatch):
    value,_,_=scene; models,calls=setup(scene,monkeypatch)
    old=(m.owner.TAG,m.owner.protocol,m.owner.verify,m.owner.EXPECTED.copy())
    out=tmp_path/'results'/(m.TAG+'_RAM')
    r=m.run_cuda(value,out,True)
    assert r['status']=='all10_raster_encodes_recorded' and r['errors']==r['finalization_errors']==[]
    assert len(calls)==10  # All input RGBs identical; still ten actual hook returns.
    assert len(r['cross_condition'])==16 and len(r['fixed_within_state_raster_difference'])==5
    assert all(x.removed==2 and not x.pre and not x.post for x in models.values())
    receipt=m.consume(value,out/'report.json',m.sha(out/'report.json'))
    assert receipt['status']=='all10_saved_raster_encodes_and16_cross_scores_verified'
    assert old==(m.owner.TAG,m.owner.protocol,m.owner.verify,m.owner.EXPECTED)


@pytest.mark.parametrize('fault',['forward','model','forbidden','replay'])
def test_failed_attempt_no_refund_full_denominators_and_cleanup(scene,tmp_path,monkeypatch,fault):
    value,_,_=scene; models,_=setup(scene,monkeypatch,fault)
    out=tmp_path/'results'/(m.TAG+'_failure'); r=m.run_cuda(value,out,True)
    assert r['status']=='failed' and len(r['items'])==10 and len(r['cross_condition'])==16
    assert all(x.removed==2 for x in models.values())
    if fault=='forward':
        assert r['meter']['attempted']['official_encoder']==1 and r['meter']['completed']['official_encoder']==0
        assert all(row['full_patch_goal_mse'] is None for row in r['cross_condition'])
    with pytest.raises(ValueError): m.consume(value,out/'report.json',m.sha(out/'report.json'))


@pytest.mark.parametrize('fault',['token_sha','token_bytes','token_path','missing','score','coverage','model','meter','seal','journal','extra','wall','science','source'])
def test_saved_consumer_mutations(scene,tmp_path,monkeypatch,fault):
    value,source,_=scene; setup(scene,monkeypatch)
    out=tmp_path/'results'/(m.TAG+'_mutation'); r=m.run_cuda(value,out,True); path=out/'report.json'
    key=m.KEYS[0]
    if fault=='token_sha': r['token_artifacts'][key]['sha256']='0'*64
    elif fault=='token_bytes': r['token_artifacts'][key]['bytes']+=1
    elif fault=='token_path': r['token_artifacts'][key]['path']=r['token_artifacts'][m.KEYS[1]]['path']
    elif fault=='missing': r['items'].pop()
    elif fault=='score': r['cross_condition'][0]['full_patch_goal_mse']+=1
    elif fault=='coverage': r['legacy_native_coverage'][0]['coverage']+=1
    elif fault=='model': r['model_after']={}
    elif fault=='meter': r['meter']['attempted']['official_encoder']=9
    elif fault=='seal': r['all10_tokens_sealed_before_scores']=False
    elif fault=='journal':
        p=out/'event_0030.json'; e=m.read_json(p); e['value']['order'].reverse(); p.write_text(json.dumps(e))
    elif fault=='extra': (out/'extra').write_text('preserved negative')
    elif fault=='wall': r['wall_seconds']=m.WALL
    elif fault=='science': r['scientific_admission']=True
    else: source.write_text('# drift\n')
    path.write_text(json.dumps(r))
    with pytest.raises((ValueError,KeyError)): m.consume(value,path,m.sha(path))


def test_scores_fixed_cartesian_product_and_full_means(monkeypatch):
    monkeypatch.setattr(m,'SHAPE',(2,3))
    features={k:np.full(m.SHAPE,i,np.float32) for i,k in enumerate(m.KEYS)}
    result=m.scores(features)
    assert [(x['word'],x['terminal_raster'],x['goal_raster']) for x in result['cross_condition']]==[
        (w,t,g) for w in m.WORDS for t in m.RASTERS for g in m.RASTERS]
    for row in result['cross_condition']:
        i=m.KEYS.index(row['word']+'__'+row['terminal_raster']); j=m.KEYS.index('goal__'+row['goal_raster'])
        assert row['full_patch_goal_mse']==(i-j)**2
    assert all(x['visible_white_feature_mse']==1 for x in result['fixed_within_state_raster_difference'])
    with pytest.raises(ValueError): m.scores(dict(reversed(list(features.items()))))


def test_predispatch_order_batch_caps_and_confirmation(scene):
    x=m.Meter(lambda *args:None); x.item=m.KEYS[0]
    with pytest.raises(ValueError): x.before('official_encoder',10)
    with pytest.raises(ValueError): x.before('predictor',1)
    x.before('official_encoder',1)
    with pytest.raises(ValueError): x.before('official_encoder',1)
    x.after('official_encoder',1)
    with pytest.raises(ValueError): x.before('official_encoder',1)  # replay image zero
    assert x.attempted['official_encoder']==1
    with pytest.raises(ValueError,match='confirmation'): m.run_cuda(scene[0],None,False)
