"""RAM fixtures only. Fake hooks are not CUDA/model evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_same_time_frame20_20261028_v4 as m


def inputs():
    features={k:np.full(m.SHAPE,i,np.float32) for i,k in enumerate(m.KEYS)}
    predictions={w:np.full(m.SHAPE,3-i,np.float32) for i,w in enumerate(m.WORDS)}
    native=[dict(word=w,frame=20,coverage=float(i),reward=float(i)/2) for i,w in enumerate(m.WORDS)]
    return features,native,predictions


def test_fixed_contract():
    p=m.protocol()
    assert p['image_order']==[w+'__frame20' for w in m.WORDS]+['goal__native']
    assert p['images']==5 and p['word_denominator']==4 and p['frame']==20
    assert p['physics']==p['AD']==p['predictor']==0
    assert p['color_intervention_frame20'] is None
    assert p['scientific_admission'] is p['C2_established'] is False


def test_all_four_and_bidirectional_ranks():
    f,n,p=inputs();r=m.scores(f,n,p)
    assert len(r['cross_condition'])==4 and len(r['same_time_rankings'])==12
    assert [x['full_patch_goal_mse'] for x in r['cross_condition']]==[16.,9.,4.,1.]
    pairs={(x['left'],x['right']):x for x in r['same_time_rankings']}
    for (a,b),row in pairs.items():
        rev=pairs[b,a]
        assert row['latent_cost_delta']==-rev['latent_cost_delta']
        assert row['coverage']['relation']=='agreement'
        assert row['coverage']['predicted_relation']=='reversal'
        assert row['reward']['native_preference']==-rev['reward']['native_preference']


def test_exact_tie_no_tolerance():
    f,n,p=inputs();n[1]['coverage']=n[0]['coverage']
    r=m.scores(f,n,p)['same_time_rankings'][0]
    assert r['coverage']['relation']==r['coverage']['predicted_relation']=='tie'
    assert m.direction(np.nextafter(0.,1.))==1
    assert m.direction(np.nextafter(0.,-1.))==-1


@pytest.mark.parametrize('kind',['missing_feature','reverse','dtype','shape','nan','missing_prediction','native_order'])
def test_invalid_denominator_abi(kind):
    f,n,p=inputs()
    if kind=='missing_feature':f.pop(m.KEYS[0])
    if kind=='reverse':f=dict(reversed(list(f.items())))
    if kind=='dtype':f[m.KEYS[0]]=f[m.KEYS[0]].astype(np.float64)
    if kind=='shape':f[m.KEYS[0]]=np.zeros((192,),np.float32)
    if kind=='nan':f[m.KEYS[0]][0,0]=np.nan
    if kind=='missing_prediction':p.pop(m.WORDS[0])
    if kind=='native_order':n.reverse()
    with pytest.raises(ValueError):m.scores(f,n,p)


def test_actual_meter_and_cap():
    events=[];meter=m.Meter(lambda *v:events.append(v))
    for key in m.KEYS:
        meter.item=key;meter.before('official_encoder',1);meter.after('official_encoder',1)
    assert meter.attempted==meter.completed==m.EXPECTED
    assert len(events)==10
    with pytest.raises(ValueError):meter.before('official_encoder',1)


@pytest.mark.parametrize('kind',['predictor','proprio_encoder','batch2','wrong_order','double_intent','return_without_intent'])
def test_meter_rejects_extra_or_wrong_calls(kind):
    meter=m.Meter(lambda *v:None);meter.item=m.KEYS[0]
    with pytest.raises(ValueError):
        if kind in ('predictor','proprio_encoder'):meter.before(kind,1)
        elif kind=='batch2':meter.before('official_encoder',2)
        elif kind=='wrong_order':meter.item=m.KEYS[1];meter.before('official_encoder',1)
        elif kind=='double_intent':meter.before('official_encoder',1);meter.before('official_encoder',1)
        else:meter.after('official_encoder',1)


def test_no_confirmation_no_preflight_or_runtime(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *x:pytest.fail('not authorized'))
    with pytest.raises(ValueError):m.run_cuda({},'/tmp/not-authorized',False)


@pytest.mark.parametrize('value',[None,'future','a'*63,'A'*64])
def test_external_sha_required(value):
    with pytest.raises(ValueError):m.actual_sha(value)


class FakeModule:
    def __init__(self):self.pre=[];self.post=[]
    def register_forward_pre_hook(self,f):
        self.pre.append(f);return SimpleNamespace(remove=lambda:self.pre.remove(f))
    def register_forward_hook(self,f):
        self.post.append(f);return SimpleNamespace(remove=lambda:self.post.remove(f))


@pytest.fixture
def ram_run(tmp_path,monkeypatch):
    root=tmp_path/'root';(root/'results').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.delenv('PYTHONPATH',raising=False)
    f,n,p=inputs();items=[];predictions={}
    for i,key in enumerate(m.KEYS):
        path=root/(key+'.npy');np.save(path,np.full((224,224,3),i,np.uint8))
        items.append(dict(image_key=key,rgb=dict(path=str(path),sha256=m.sha(path),bytes=path.stat().st_size)))
    for w in m.WORDS:
        path=root/(w+'_prediction.npy');np.save(path,np.stack([p[w]]*3))
        predictions[w]=dict(path=str(path),sha256=m.sha(path),bytes=path.stat().st_size)
    manifest=dict(inputs=items,legacy_native_coverage=n,predictions=predictions,
        model_spec=dict(loader_manifest=dict(expected_model_receipts={'fixture':'RAM only'})),
        missing=[dict(word=w,frame=20,white_rgb=None) for w in m.WORDS],goal_first_stage='null_pixel_first_stage_no_encoder_invariance_test')
    modules={k:FakeModule() for k in m.EXPECTED}
    monkeypatch.setattr(m,'verify',lambda *x:None)
    monkeypatch.setattr(m,'source_closure',lambda *x:None)
    monkeypatch.setattr(m,'load_runtime',lambda *x:(modules,{k:dict(exact_match=True) for k in ('predictor','proprio_encoder')},
        {},{'fixture':'RAM only'},lambda:{'fixture':'RAM only'},lambda:{}))
    def fake_encode(module,rgb):
        args=(SimpleNamespace(is_cuda=True,shape=(1,3,224,224)),)
        for h in module.pre:h(module,args)
        out=np.full(m.SHAPE,int(rgb[0,0,0]),np.float32)
        for h in module.post:h(module,args,out)
        return out
    monkeypatch.setattr(m,'encode',fake_encode)
    output=root/'results'/(m.TAG+'_fixture')
    report=m.run_cuda(manifest,output,True)
    return manifest,output,report


def test_ram_full_event_store_score_consume(ram_run):
    manifest,out,r=ram_run
    assert r['status']=='all5_raster_encodes_recorded'
    receipt=m.consume(manifest,out/'report.json',m.sha(out/'report.json'))
    assert receipt['status']=='all5_frame20_encodes_and4_scores_12_rankings_verified'
    assert len(receipt['same_time_rankings'])==12
    assert len(list(out.glob('event_*.json')))==16


def test_failure_keeps_four_scores_and_missing_cells(ram_run,monkeypatch):
    manifest,out,_=ram_run
    def failure(*args):raise RuntimeError('fixture encoder failed; no retry')
    monkeypatch.setattr(m,'encode',failure)
    target=out.parent/(m.TAG+'_failure_fixture')
    report=m.run_cuda(manifest,target,True)
    assert report['status']=='failed' and report['all5_tokens_sealed_before_scores'] is False
    assert len(report['items'])==5 and len(report['cross_condition'])==4
    assert report['missing']==manifest['missing'] and len(report['missing'])==4
    assert all(row['full_patch_goal_mse'] is None for row in report['cross_condition'])
    with pytest.raises(ValueError):m.consume(manifest,target/'report.json',m.sha(target/'report.json'))


@pytest.mark.parametrize('mutation',['score','rank','meter','seal','artifact','extra_file','color_imputation'])
def test_saved_mutation_rejected(ram_run,mutation):
    manifest,out,r=ram_run;report=out/'report.json'
    if mutation=='extra_file':(out/'unexpected').touch()
    elif mutation=='seal':
        event=out/'event_0015.json';x=m.read_json(event);x['kind']='early_scored';event.write_text(__import__('json').dumps(x))
    elif mutation=='artifact':
        a=out/(m.KEYS[0]+'_patch.npy');np.save(a,np.ones(m.SHAPE,np.float32))
    else:
        x=deepcopy(r)
        if mutation=='score':x['cross_condition'][0]['full_patch_goal_mse']+=1
        if mutation=='rank':x['same_time_rankings'][0]['coverage']['relation']='invented'
        if mutation=='meter':x['meter']['frames_completed']['official_encoder']=4
        if mutation=='color_imputation':x['missing'][0]['white_rgb']='fake.png'
        report.write_text(__import__('json').dumps(x))
    with pytest.raises(ValueError):m.consume(manifest,report,m.sha(report))
