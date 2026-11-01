"""RAM token recorders only, never neural/physics/scientific evidence."""
from copy import deepcopy
import ast
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_contact_action_response_20261028_v2 as m


STATS = dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002],
    PROPRIO_MEAN=[236.6155,264.5674,-2.93032027,2.54307914],
    PROPRIO_STD=[101.1202,87.0112,74.84556075,74.14009094])


def npy(path,a):
    with path.open('xb') as f:
        np.save(f,a,allow_pickle=False)
    return dict(path=str(path),sha256=m.sha(path),bytes=path.stat().st_size,
                shape=list(a.shape),dtype=str(a.dtype))


@pytest.fixture
def scene(tmp_path,monkeypatch):
    """Public-only fabricated fixture; no source/physics private directories."""
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'PATCHES',2)
    monkeypatch.setattr(m,'FEATURES',3)
    monkeypatch.setattr(m,'PROP',2)
    monkeypatch.setattr(m,'FLOOR',0)
    (tmp_path/'results').mkdir()
    public = tmp_path/'public'
    public.mkdir()
    data = {**{f'context_rgb_{i:02d}':np.full((224,224,3),i,np.uint8) for i in (0,5,10)},
        'goal_rgb':np.full((224,224,3),99,np.uint8),
        'context_proprio':np.arange(12,dtype=np.float32).reshape(3,4),
        'history_raw_u':np.zeros((10,2),np.float64),
        'history_normalized':m.owner.normalize(np.zeros((2,5,2),np.float64),STATS)}
    records = {k:npy(public/(k+'.npy'),v) for k,v in data.items()}
    payload = tmp_path/'payload.json'
    m.write_new(payload,dict(schema=m.PUBLIC_SCHEMA,inputs=records,proprio_units=m.PUBLIC_UNITS,
        ready_for_numeric_context_loading=True,source_witness_included=False,scientific_admission=False))
    monkeypatch.setattr(m,'PUBLIC_SHA',m.sha(payload))
    stub = tmp_path/'model_source.py'
    stub.write_text('# inert authority fixture, not a model\n')
    spec = dict(loader_manifest=dict(seals={str(stub):m.sha(stub)},expected_model_receipts={'fixture':True}),stats=STATS)
    monkeypatch.setattr(m,'MODEL_SPEC_SHA',m.digest(spec))
    monkeypatch.setattr(m,'model_authority',lambda:deepcopy(spec))
    monkeypatch.setattr(m,'preprocessing_pairing',lambda spec:dict(fixture_only=True,stats=deepcopy(spec['stats'])))
    monkeypatch.setattr(m,'self_bindings',lambda:{str(stub):m.sha(stub)})
    prepared = m.prepare(payload,m.sha(payload))
    return prepared,payload,data,records


def fixture_run(prepared,directory):
    """Analytic constant recorder behind the actual saved layout/causal seam."""
    arrays = m.verify(prepared)
    store = m.Store(directory)
    meter = m.Meter(store.event,store.guard)
    records = {}
    def emit(w,s,n,a):
        records[f'{w}_{s}_{n}']=store.emit(w,s,n,a)
    meter.before('official_encoder',3)
    visual = np.arange(18,dtype=np.float32).reshape(3,2,3)
    meter.after('official_encoder',3)
    emit('shared',-1,'visual_features',visual)
    emit('shared',-1,'normalized_proprio',m.normalized_proprio(arrays['context_proprio'],STATS))
    meter.before('proprio_encoder',1)
    prop = np.arange(12,dtype=np.float32).reshape(3,2,2)
    meter.after('proprio_encoder',1)
    emit('shared',-1,'proprio_features',prop)
    def predict(v,a,p):
        meter.before('predictor',1)
        out = v+np.float32(a[-1].sum()),p+np.float32(.25)
        meter.after('predictor',1)
        return out
    terminal = m.rollout(predict,visual,prop,arrays['history_normalized'],
        {k:m.owner.normalize(v,STATS) for k,v in m.programs().items()},emit)
    r = dict(schema=m.TAG,status='public_native_action_response_recorded',manifest_sha256=m.digest(prepared),
        protocol=m.protocol(),words=[dict(word=w,status='completed',error=None) for w in m.WORDS],
        artifacts=records,contrasts=m.contrasts(terminal),meter=meter.snapshot(),
        errors=[],finalization_errors=[],cuda_executed=True,scientific_admission=False,physical_release=False,
        model_before={'fixture':True},model_after={'fixture':True},wall_seconds=.1,journal_terminal_sha256=store.previous)
    path = store.output/'report.json'
    m.write_new(path,r)
    return path,r


def rewrite(path,value):
    # Test-only mutation of private temporary fixtures, never MAIN files.
    path.write_text(json.dumps(value,allow_nan=False))


def test_protocol_and_original_programs():
    p = m.protocol()
    assert p['words']==['push_x','raw_zero'] and p['denominator']==2
    assert p['counts_api']==dict(official_encoder=1,proprio_encoder=1,predictor=4)
    assert p['counts_forward_samples']==dict(official_encoder=3,proprio_encoder=1,predictor=4)
    assert p['counts_context_frames']==dict(official_encoder=3,proprio_encoder=3,predictor=12)
    assert p['FLOPs'] is None and p['task_success'] is None
    assert not any(p[k] for k in ('scientific_admission','fresh','goal_encoded','goal_scoring','physics','AD','JVP'))
    assert m.WORDS==('push_x','raw_zero')
    assert m.programs()['push_x'].dtype==np.float64
    assert m.owner.normalize(m.programs()['raw_zero'],STATS).any()


def test_saved_prepare_verify_public_only(scene):
    prepared,path,data,_ = scene
    arrays = m.verify(prepared)
    assert set(arrays)==set(data)
    text = json.dumps(prepared)
    assert not any(s in text for s in ('full_public_state','contacts','reference_suffix','generator_private','publication_private'))
    assert prepared==m.prepare(path,m.sha(path))


def test_wrong_external_public_sha(scene):
    _,p,_,_ = scene
    with pytest.raises(ValueError,match='actual public'):
        m.prepare(p,'0'*64)


@pytest.mark.parametrize('mutation',['private_key','scientific','witness','not_ready','units','schema','extra_input'])
def test_public_json_mutations(scene,monkeypatch,mutation):
    _,path,_,_ = scene
    v = m.read_json(path)
    if mutation=='private_key': v['private_lineage']='forbidden'
    elif mutation=='scientific': v['scientific_admission']=True
    elif mutation=='witness': v['source_witness_included']=True
    elif mutation=='not_ready': v['ready_for_numeric_context_loading']=False
    elif mutation=='units': v['proprio_units']='block velocities'
    elif mutation=='schema': v['schema']='numeric_projection_only'
    elif mutation=='extra_input': v['inputs']['suffix']=v['inputs']['history_raw_u']
    rewrite(path,v)
    monkeypatch.setattr(m,'PUBLIC_SHA',m.sha(path))
    with pytest.raises(ValueError):
        m.prepare(path,m.sha(path))


@pytest.mark.parametrize('mutation',['extra_file','symlink','normalized_zero','dtype','nan','shape','path','bytes','raw_nonzero'])
def test_public_array_mutations(scene,monkeypatch,mutation):
    prepared,path,data,records = scene
    v = m.read_json(path)
    directory = Path(records['history_raw_u']['path']).parent
    if mutation=='extra_file':
        (directory/'private.json').write_text('{}')
    elif mutation=='symlink':
        target = directory/'history_raw_u.npy'
        target.rename(directory.parent/'retained_raw.npy')
        target.symlink_to(directory.parent/'retained_raw.npy')
    elif mutation=='path':
        v['inputs']['context_rgb_00']['path']=records['goal_rgb']['path']
    elif mutation=='bytes':
        v['inputs']['context_proprio']['bytes']+=1
    else:
        k = 'history_normalized' if mutation=='normalized_zero' else 'context_proprio'
        if mutation=='raw_nonzero': k='history_raw_u'
        a = data[k].copy()
        if mutation=='normalized_zero': a[:]=0
        if mutation=='dtype': a=a.astype(np.float64)
        if mutation=='nan': a[0,0]=np.nan
        if mutation=='shape': a=a[:2]
        if mutation=='raw_nonzero': a[0,0]=.1
        target = directory/(k+'.npy')
        with target.open('wb') as f: np.save(f,a,allow_pickle=False)
        v['inputs'][k]=dict(path=str(target),sha256=m.sha(target),bytes=target.stat().st_size,
                           shape=list(a.shape),dtype=str(a.dtype))
    rewrite(path,v)
    monkeypatch.setattr(m,'PUBLIC_SHA',m.sha(path))
    with pytest.raises(ValueError):
        m.prepare(path,m.sha(path))


def test_proprio_original_float32_and_not_refit():
    a = np.arange(12,dtype=np.float32).reshape(3,4)
    out = m.normalized_proprio(a,STATS)
    assert np.array_equal(out,(a-np.asarray(STATS['PROPRIO_MEAN'],np.float32))/np.asarray(STATS['PROPRIO_STD'],np.float32))
    with pytest.raises(ValueError): m.normalized_proprio(a.astype(np.float64),STATS)
    bad = deepcopy(STATS); bad['PROPRIO_STD'][0]=0
    with pytest.raises(ValueError): m.normalized_proprio(a,bad)


@pytest.mark.parametrize('mutation',['words','budget','stats','model','extra_private','self','scientific'])
def test_prepared_mutations(scene,mutation):
    original,_,_,_ = scene
    v = deepcopy(original)
    if mutation=='words': v['plans'].pop('push_x')
    elif mutation=='budget': v['protocol']['counts_api']['predictor']=9
    elif mutation=='stats': v['model_spec']['stats']['ACTION_MEAN'][0]=0
    elif mutation=='model': v['model_spec']['loader_manifest']['expected_model_receipts']={}
    elif mutation=='extra_private': v['private_lineage']='forbidden'
    elif mutation=='self': v['bindings']={}
    elif mutation=='scientific': v['scientific_admission']=True
    with pytest.raises(ValueError): m.verify(v)


def test_meter_samples_not_api_and_failure_not_refunded():
    events=[]
    meter = m.Meter(lambda k,v:events.append((k,v)))
    meter.before('official_encoder',3)
    assert meter.snapshot()['attempted']['official_encoder']==1
    assert meter.snapshot()['samples_attempted']['official_encoder']==3
    assert meter.completed['official_encoder']==0
    with pytest.raises(ValueError): meter.before('official_encoder',3)
    meter.after('official_encoder',3)
    with pytest.raises(ValueError): meter.before('official_encoder',3)
    with pytest.raises(ValueError): meter.after('official_encoder',3)
    assert len(events)==2


@pytest.mark.parametrize('key,batch',[('official_encoder',1),('predictor',2),('proprio_encoder',3),('predictor',True),('unknown',1)])
def test_predispatch_invalid_batch(key,batch):
    meter=m.Meter(lambda k,v:None)
    with pytest.raises(ValueError): meter.before(key,batch)
    assert not any(meter.attempted.values())


def test_predispatch_caps_and_pairs():
    meter=m.Meter(lambda k,v:None)
    for _ in range(4):
        meter.before('predictor',1)
        with pytest.raises(ValueError): meter.after('proprio_encoder',1)
        meter.after('predictor',1)
    with pytest.raises(ValueError): meter.before('predictor',1)
    assert meter.completed['predictor']==4


def test_saved_recompute_full_denominator(scene):
    prepared,_,_,_ = scene
    path,r = fixture_run(prepared,m.ROOT/'results'/(m.TAG+'_fixture'))
    result = m.consume(prepared,path,m.sha(path))
    assert result['status']=='all2_saved_public_action_words_verified'
    assert result['scientific_admission'] is False and result['task_success'] is None
    assert result['contrasts']==r['contrasts'] and result['meter']['samples_completed']['official_encoder']==3


@pytest.mark.parametrize('mutation',['missing_word','failed','admission','model','counts','batch','contrast','wall','journal','array','extra'])
def test_saved_report_mutations(scene,mutation):
    prepared,_,_,_ = scene
    path,r = fixture_run(prepared,m.ROOT/'results'/(m.TAG+'_fixture'))
    if mutation=='missing_word': r['words']=r['words'][:-1]
    elif mutation=='failed': r['status']='failed'
    elif mutation=='admission': r['scientific_admission']=True
    elif mutation=='model': r['model_after']={}
    elif mutation=='counts': r['meter']['completed']['predictor']=7
    elif mutation=='batch': r['meter']['samples_completed']['official_encoder']=1
    elif mutation=='contrast': r['contrasts']['push_vs_raw_zero_patch_mse']+=1
    elif mutation=='wall': r['wall_seconds']=m.WALL+1
    elif mutation=='journal': r['journal_terminal_sha256']='0'*64
    elif mutation=='array':
        rec = r['artifacts']['push_x_0_actions_in']
        p = Path(rec['path'])
        a = np.load(p,allow_pickle=False); a[0,0]+=1
        with p.open('wb') as f: np.save(f,a,allow_pickle=False)
    elif mutation=='extra': (path.parent/'unbound.npy').write_bytes(b'bad')
    rewrite(path,r)
    with pytest.raises(ValueError): m.consume(prepared,path,m.sha(path))


def test_saved_causal_mutation_resealed_hashes(scene):
    prepared,_,_,_ = scene
    path,r = fixture_run(prepared,m.ROOT/'results'/(m.TAG+'_fixture'))
    key='raw_zero_1_proprio_in'
    rec=r['artifacts'][key]
    p=Path(rec['path']); a=np.load(p,allow_pickle=False); a[0,0,0]+=1
    with p.open('wb') as f: np.save(f,a,allow_pickle=False)
    rec['sha256']=m.sha(p)
    previous=None
    for epath in sorted(path.parent.glob('event_*.json')):
        e=m.read_json(epath)
        if e['kind']=='artifact' and e['value']['key']==key: e['value']['record']=rec
        e['previous_sha256']=previous
        e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']; rewrite(epath,e)
    r['journal_terminal_sha256']=previous; rewrite(path,r)
    with pytest.raises(ValueError,match='causal'): m.consume(prepared,path,m.sha(path))


def test_no_top_level_torch_physics_or_network():
    tree=ast.parse(Path(m.__file__).read_text())
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            assert not any(s.split('.')[0] in ('torch','pymunk','requests','urllib','pickle') for s in names)
    assert 'project_blind' not in Path(m.__file__).read_text()


def test_real_saved_projection_and_public_only():
    """Only fixed JSON + seven public NPY; no checkpoint reads/import/load/prepare."""
    actual = m.read_json(m.owner.MANIFEST,m.owner.PINS[str(m.owner.MANIFEST)])
    report = m.read_json(m.owner.REPORT,m.owner.PINS[str(m.owner.REPORT)])
    m.owner.verify_v12_manifest_digest(actual,report)
    spec=m.model_projection(actual)
    pairing=m.preprocessing_pairing(spec)
    assert pairing['checkpoint']['sha256']=='9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb'
    assert pairing['stats']==STATS
    assert pairing['inference_crop_scale']==pairing['training_crop_scale']==[1.777,1.777]
    assert m.digest(spec)==m.MODEL_SPEC_SHA
    assert spec['stats']==STATS
    assert not any('/results/' in p or '/pusht_noise/' in p for p in spec['loader_manifest']['seals'])
    assert m.sha(m.owner.__file__)==m.OWNER_SHA
    p,arrays=m.public_inputs(m.PUBLIC,m.PUBLIC_SHA,spec['stats'])
    assert len(p['inputs'])==7
    assert arrays['context_proprio'].shape==(3,4)


def test_native_crop_source_square_pairing_no_torch():
    """Execute ONLY extracted pure crop-coordinate helper, never model/transform."""
    source=m.ROOT/'external/jepa-wms/src/datasets/utils/video/transforms.py'
    assert m.sha(source)==m.PAIR_PINS['src/datasets/utils/video/transforms.py']
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_get_param_spatial_crop')
    tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace={'random':random.Random(101),'np':SimpleNamespace(random=SimpleNamespace(uniform=lambda:.75)),'math':math}
    exec(compile(tree,str(source),'exec'),namespace)
    crop=namespace['_get_param_spatial_crop']
    assert crop((1.777,1.777),(1.,1.),224,224)==(0,0,224,224)
    assert crop((1.,1.),(1.,1.),224,224)==(0,0,224,224)
    # Adapter actual make_transforms literal parameters pair with the YAML.
    tree=ast.parse(Path(m.__file__).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_transforms']
    assert len(calls)==1
    params={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
    assert params==dict(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1.,1.),
        random_resize_scale=(1.777,1.777),reprob=0.,auto_augment=False,motion_shift=False)


def test_original_two_arm_numeric_protocol_no_private_selection():
    plans=m.programs()
    assert tuple(plans)==('push_x','raw_zero')
    assert plans['push_x'].shape==(2,5,2) and plans['push_x'].dtype==np.float64
    assert np.array_equal(plans['push_x'][...,0],np.full((2,5),.1,np.float64))
    assert not plans['push_x'][...,1].any() and not plans['raw_zero'].any()
    assert np.array_equal(m.owner.normalize(plans['raw_zero'],STATS),
        np.tile((-np.asarray(STATS['ACTION_MEAN'],np.float32)/np.asarray(STATS['ACTION_STD'],np.float32)),(2,5)).reshape(2,10))


def test_four_word_v1_remains_frozen():
    stem='agent_p5_contact_action_response_20261028_v1'
    assert [m.sha(p) for p in (m.ROOT/'experiments/spectral_reachability'/(stem+'.py'),
        m.ROOT/'tests'/('test_'+stem+'.py'),m.ROOT/'innerdocs'/(stem+'.md'))]==[
        'bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930',
        'bf11d51657c1a80413b5cd842c010e955195dacc65b942a662be692d64b5965d',
        '38734c5b0cfd01ecf78b066f65cbf9bd30260ab3d0ab8c2f1247a14eb8f0a523']


def test_store_guards_before_writes(scene,monkeypatch):
    store=m.Store(m.ROOT/'results'/(m.TAG+'_guards'))
    monkeypatch.setattr(m,'CAP',1024)
    with pytest.raises(ValueError,match='output cap'): store.emit('push_x',0,'visual_in',np.zeros((3,2,3),np.float32))
    assert not list(store.output.iterdir())


@pytest.mark.parametrize('fail_at',[None,'official_encoder','predictor'])
def test_run_lifecycle_record_only_hooks_cleanup(scene,monkeypatch,fail_at):
    """No tensor/model implementation: hooks around inert Python tokens only."""
    prepared,_,_,_ = scene
    receipts={k:{'token':k} for k in m.EXPECTED}
    prepared['model_spec']['loader_manifest']['expected_model_receipts']=receipts
    monkeypatch.setattr(m,'MODEL_SPEC_SHA',m.digest(prepared['model_spec']))
    removed=[]
    class TokenModule:
        def __init__(self,key): self.key,self.pre,self.post=key,[],[]
        def register_forward_pre_hook(self,h):
            self.pre.append(h)
            return SimpleNamespace(remove=lambda:removed.append((self.key,'pre')))
        def register_forward_hook(self,h):
            self.post.append(h)
            return SimpleNamespace(remove=lambda:removed.append((self.key,'post')))
        def call(self):
            shape={'official_encoder':(3,3,224,224),'proprio_encoder':(1,3,4),
                   'predictor':(1,3,1,16,16,384)}[self.key]
            args=(SimpleNamespace(is_cuda=True,shape=shape),)
            for h in self.pre: h(self,args)
            if self.key==fail_at: raise RuntimeError('injected token exception')
            for h in self.post: h(self,args,None)
    models={k:TokenModule(k) for k in m.EXPECTED}
    receipt=SimpleNamespace(model_receipt=lambda v:receipts[v.key])
    loader=SimpleNamespace(__file__=next(iter(prepared['bindings'])),
        load_cuda=lambda root,manifest:(models,{'token':True},{},receipts),
        helper=lambda root:SimpleNamespace(dependencies=lambda root:(None,
            SimpleNamespace(import_v4=lambda root:receipt),None)))
    old_import=m.importlib.import_module
    monkeypatch.setattr(m.importlib,'import_module',lambda name:loader if name.endswith(m.owner.LOADER) else old_import(name))
    monkeypatch.setattr(m.owner,'LOADER_SHA',m.sha(loader.__file__))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:True,mem_get_info=lambda:(8*1024**3,16*1024**3))))
    monkeypatch.delenv('PYTHONPATH',raising=False)
    def fake_case(models,arrays,stats,emit):
        models['official_encoder'].call()
        v=np.zeros((3,2,3),np.float32)
        emit('shared',-1,'visual_features',v)
        emit('shared',-1,'normalized_proprio',m.normalized_proprio(arrays['context_proprio'],stats))
        models['proprio_encoder'].call()
        p=np.zeros((3,2,2),np.float32)
        emit('shared',-1,'proprio_features',p)
        def predict(v,a,p):
            models['predictor'].call()
            return v+np.float32(1),p+np.float32(1)
        return m.contrasts(m.rollout(predict,v,p,arrays['history_normalized'],
            {k:m.owner.normalize(a,stats) for k,a in m.programs().items()},emit))
    monkeypatch.setattr(m,'cuda_case',fake_case)
    directory=m.ROOT/'results'/(m.TAG+'_lifecycle')
    r=m.run_cuda(prepared,directory)
    assert len(removed)==6 and len(r['words'])==2
    assert r['model_before']==r['model_after']==receipts
    assert m.read_json(directory/'report.json')==r
    if fail_at is None:
        assert m.consume(prepared,directory/'report.json',m.sha(directory/'report.json'))['status']=='all2_saved_public_action_words_verified'
    else:
        assert r['status']=='failed' and r['errors'] and not r['finalization_errors']
        assert r['meter']['pending']==fail_at
        assert r['meter']['attempted'][fail_at]==1 and r['meter']['completed'][fail_at]==0
        with pytest.raises(ValueError): m.consume(prepared,directory/'report.json',m.sha(directory/'report.json'))


def test_cli_import_does_not_load_torch():
    code=f'import sys; from experiments.spectral_reachability import {m.TAG}; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-B','-c',code],check=True,cwd=m.ROOT)


def test_original_native_frozen_three():
    stem='agent_p5_contact_source_native_20261028_v1'
    paths=[m.ROOT/'experiments/spectral_reachability'/(stem+'.py'),m.ROOT/'tests'/('test_'+stem+'.py'),m.ROOT/'innerdocs'/(stem+'.md')]
    assert [m.sha(p) for p in paths]==[
        '78700888a70d5caf263d04301ca8a62f907364ba4b38cf9d3bdaf689d115fecd',
        '0095356147503a2150da431560be33e220df4e70870160a6a66cee07ef050f38',
        'b0a6159dc6c6f287a095261ccbcd073a8a40b67c21715b5d13d844dd5a4a23ec']

