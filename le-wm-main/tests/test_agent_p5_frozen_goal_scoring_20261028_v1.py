"""RAM receipt/token fixtures only, no neural/physics or scientific evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_frozen_goal_scoring_20261028_v1 as m


def npy(path,a):
    with path.open('xb') as f: np.save(f,a,allow_pickle=False)
    return dict(path=str(path),sha256=m.sha(path),bytes=path.stat().st_size,shape=list(a.shape),dtype=str(a.dtype))


def rewrite(path,v): path.write_text(json.dumps(v,allow_nan=False))


@pytest.fixture
def scene(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); monkeypatch.setattr(m,'SHAPE',(2,3)); monkeypatch.setattr(m,'FLOOR',0)
    (tmp_path/'results').mkdir()
    source=tmp_path/'source.py'; source.write_text('# inert RAM source\n')
    receipt=tmp_path/'prediction_receipt.json'
    m.write_new(receipt,dict(meter={'RAM_fixture':True}))
    monkeypatch.setattr(m,'RECEIPT_SHA',m.sha(receipt))
    expected={k:{'RAM_fixture':True} for k in m.EXPECTED}
    mm=dict(public_payload_sha256='1'*64,inputs=dict(goal_rgb={'RAM_fixture':True}),
        model_spec=dict(loader_manifest=dict(expected_model_receipts=expected,seals={str(source):m.sha(source)})))
    terminals={w:npy(tmp_path/(w+'.npy'),np.full((3,2,3),i,np.float32)) for i,w in enumerate(m.WORDS)}
    plan=dict(public_manifest=mm,terminals=terminals,authority=dict(receipt_path=str(receipt)))
    monkeypatch.setattr(m,'verify',lambda plan:dict(goal_rgb=np.zeros((224,224,3),np.uint8)))
    return plan,source


def fixture_run(plan,out,source):
    store=m.Store(out); meter=m.Meter(store.event,store.guard)
    meter.before('official_encoder',1); meter.after('official_encoder',1)
    rec=store.goal(np.zeros(m.SHAPE,np.float32))
    before=plan['public_manifest']['model_spec']['loader_manifest']['expected_model_receipts']
    store.json('goal_encoding.json',m.seal_value(plan,rec,before))
    store.event('goal_sealed',dict(sha256=m.sha(store.out/'goal_encoding.json'),record=rec))
    result=m.score_after_seal(plan,store,rec,before)
    store.event('scored',result)
    r=dict(schema=m.TAG+'.report',status='all4_public_goal_scores_recorded',manifest_sha256=m.digest(plan),
        protocol=m.protocol(),words=result['rows'],selected_word=result['selected_word'],
        errors=[],finalization_errors=[],cuda_executed=True,scientific_admission=False,C2_established=False,
        physical_release=False,task_success=None,model_before=before,model_after=before,
        strict_loads={k:dict(exact_match=True) for k in ('predictor','proprio_encoder')},
        imported_sources={'fixture':str(source)},wall_seconds=.01,goal_patch=rec,
        goal_encoding_sha256=result['goal_encoding_sha256'],journal_terminal_sha256=store.previous,meter=meter.snapshot())
    path=store.out/'report.json'; m.write_new(path,r)
    return path,r


def test_scoring_and_exact_tie_order():
    goal=np.zeros(m.SHAPE,np.float32)
    t={w:np.full(m.SHAPE,i+1,np.float32) for i,w in enumerate(m.WORDS)}
    r=m.scores(goal,t)
    assert [x['goal_patch_mse'] for x in r['rows']]==[1.,4.,9.,16.]
    assert r['selected_word']=='xy'
    t={w:goal.copy() for w in m.WORDS}
    assert m.scores(goal,t)['selected_word']=='xy'
    t['xy'][:]=np.float32(1e-8)
    assert m.scores(goal,t)['selected_word']=='yx'  # no epsilon/tolerance tie


@pytest.mark.parametrize('change',['missing','order','nan','float64','shape'])
def test_bad_score_inputs(change):
    goal=np.zeros(m.SHAPE,np.float32); t={w:goal.copy() for w in m.WORDS}
    if change=='missing': t.pop('xy')
    elif change=='order': t=dict(reversed(list(t.items())))
    elif change=='nan': t['xy'][0,0]=np.nan
    elif change=='float64': goal=goal.astype(np.float64)
    else: t['xy']=t['xy'][:1]
    with pytest.raises(ValueError): m.scores(goal,t)


def test_meter_no_extra_failure_refund_or_unit_sum():
    events=[]; meter=m.Meter(lambda k,v:events.append((k,v)))
    meter.before('official_encoder',1)
    assert meter.attempted['official_encoder']==1 and meter.completed['official_encoder']==0
    with pytest.raises(ValueError): meter.before('official_encoder',1)
    with pytest.raises(ValueError): meter.after('predictor',1)
    meter.after('official_encoder',1)
    for key,batch in [('official_encoder',1),('predictor',1),('proprio_encoder',1),('official_encoder',3)]:
        with pytest.raises(ValueError): meter.before(key,batch)
    assert meter.snapshot()['samples_completed']==m.EXPECTED
    assert m.protocol()['FLOPs'] is None and m.protocol()['new_api']==m.EXPECTED


def test_seal_required_before_scoring(scene,tmp_path,monkeypatch):
    plan,source=scene; store=m.Store(tmp_path/'results'/f'{m.TAG}_no_seal')
    before=plan['public_manifest']['model_spec']['loader_manifest']['expected_model_receipts']
    rec=store.goal(np.zeros(m.SHAPE,np.float32))
    monkeypatch.setattr(m,'scores',lambda *args:pytest.fail('no scoring before persisted seal'))
    with pytest.raises(ValueError,match='regular nonsymlink'): m.score_after_seal(plan,store,rec,before)
    store.json('goal_encoding.json',{'invalid':True})
    with pytest.raises(ValueError,match='durable goal'): m.score_after_seal(plan,store,rec,before)


def test_full_actual_layout_saved_consumer(scene,tmp_path):
    plan,source=scene
    path,r=fixture_run(plan,tmp_path/'results'/f'{m.TAG}_RAM',source)
    got=m.consume(plan,path,m.sha(path))
    assert got['status']=='all4_saved_goal_scores_verified' and len(got['words'])==4
    assert got['selected_word']=='xy' and got['C2_established'] is False and got['task_success'] is None
    assert got['new_meter']['completed']==m.EXPECTED
    assert got['reused_meter']=={'RAM_fixture':True}
    assert len(list(path.parent.glob('event_*.json')))==5


@pytest.mark.parametrize('change',['score','selected','missing','partial','failure','finalization','model','meter','terminal',
    'seal','array','event','extra','wall','source','strict_load','science','cpu','goal_record'])
def test_saved_tamper(scene,tmp_path,change):
    plan,source=scene; path,r=fixture_run(plan,tmp_path/'results'/f'{m.TAG}_tamper',source)
    if change=='score': r['words'][0]['goal_patch_mse']=123.
    elif change=='selected': r['selected_word']='yx'
    elif change=='missing': r['words'].pop()
    elif change=='partial': r['status']='partial'
    elif change=='failure': r['errors']=['failure']
    elif change=='finalization': r['finalization_errors']=['cleanup']
    elif change=='model': r['model_after']={}
    elif change=='meter': r['meter']['completed']['official_encoder']=0
    elif change=='terminal': r['journal_terminal_sha256']='0'*64
    elif change=='seal': rewrite(path.parent/'goal_encoding.json',{})
    elif change=='array':
        with (path.parent/'goal_patch.npy').open('wb') as f: np.save(f,np.ones(m.SHAPE,np.float32))
    elif change=='event':
        e=m.read_json(path.parent/'event_0003.json'); e['kind']='scored'; rewrite(path.parent/'event_0003.json',e)
    elif change=='extra': (path.parent/'private.json').write_text('{}')
    elif change=='wall': r['wall_seconds']=m.WALL
    elif change=='source': source.write_text('# drift\n')
    elif change=='strict_load': r['strict_loads']['predictor']['exact_match']=False
    elif change=='science': r['C2_established']=True
    elif change=='cpu': r['cuda_executed']=False
    else: r['goal_patch']['path']=str(tmp_path/'different.npy')
    rewrite(path,r)
    with pytest.raises((ValueError,KeyError)): m.consume(plan,path,m.sha(path))


def test_run_requires_explicit_main_authority(scene,tmp_path):
    with pytest.raises(ValueError,match='explicit MAIN'): m.run_cuda(scene[0],tmp_path/'unused')
    assert not (tmp_path/'unused').exists()


def test_prepare_external_actual_authority_first(monkeypatch,tmp_path):
    monkeypatch.setattr(m.base,'consume',lambda *args:pytest.fail('invalid external SHA must reject before consume'))
    with pytest.raises(ValueError,match='actual four-word'):
        m.prepare(tmp_path/'m','0'*64,tmp_path/'r',m.REPORT_SHA,tmp_path/'s',m.RECEIPT_SHA)


def test_prepare_wiring_and_receipt_mutation(scene,tmp_path,monkeypatch):
    plan,source=scene
    original_sha=m.sha
    def fake_sha(path):
        if Path(path)==Path(m.base.__file__): return m.BASE_SHA
        return original_sha(path)
    monkeypatch.setattr(m,'sha',fake_sha)
    monkeypatch.setattr(m,'NATIVE',{})
    monkeypatch.setattr(m,'self_bindings',lambda:{str(source):original_sha(source)})
    manifest=tmp_path/'saved_manifest.json'; report=tmp_path/'saved_report.json'; receipt=tmp_path/'saved_receipt.json'
    saved_receipt=dict(status='all4_saved_public_action_words_verified')
    m.write_new(manifest,plan['public_manifest'])
    m.write_new(report,dict(artifacts={w+'_1_visual_out':plan['terminals'][w] for w in m.WORDS}))
    m.write_new(receipt,saved_receipt)
    for name,path in [('MANIFEST_SHA',manifest),('REPORT_SHA',report),('RECEIPT_SHA',receipt)]:
        monkeypatch.setattr(m,name,original_sha(path))
    monkeypatch.setattr(m.base,'consume',lambda *args:saved_receipt)
    result=m.prepare(manifest,m.MANIFEST_SHA,report,m.REPORT_SHA,receipt,m.RECEIPT_SHA)
    assert result['status']=='prepared_goal_not_encoded_not_scored'
    assert set(result['terminals'])==set(m.WORDS) and result['protocol']['seal_before_score'] is True
    assert 'scores' not in result and 'selected_word' not in result
    monkeypatch.setattr(m.base,'consume',lambda *args:{})
    with pytest.raises(ValueError,match='full saved numeric strict'):
        m.prepare(manifest,m.MANIFEST_SHA,report,m.REPORT_SHA,receipt,m.RECEIPT_SHA)


def test_verify_rejects_changed_contract(monkeypatch):
    monkeypatch.setattr(m,'prepare',lambda **args:dict(authority={},schema='expected'))
    with pytest.raises(ValueError,match='manifest/source drift'): m.verify(dict(authority={},schema='changed'))


def test_original_source_and_saved_authority_no_scoring():
    assert m.sha(m.base.__file__)==m.BASE_SHA
    for rel,h in m.NATIVE.items(): assert m.sha(m.ROOT/rel)==h
    for suffix,h in [('prepare_MAIN01.json',m.MANIFEST_SHA),('cuda_MAIN01/report.json',m.REPORT_SHA),
                     ('consume_MAIN01.json',m.RECEIPT_SHA)]:
        path=m.ROOT/'results'/('agent_p5_contact_action_response_20261028_v1_'+suffix)
        assert m.sha(path)==h
    assert m.protocol()['known_single_constructed_OOD_root'] and not m.protocol()['prospective']


class TokenModel:
    """Hook-only counter, never a neural model or physical simulation."""
    def __init__(self): self.pre=[]; self.post=[]; self.removed=0
    def register_forward_pre_hook(self,h): self.pre.append(h); return self.handle(self.pre,h)
    def register_forward_hook(self,h): self.post.append(h); return self.handle(self.post,h)
    def handle(self,bank,h):
        def remove(): bank.remove(h); self.removed+=1
        return SimpleNamespace(remove=remove)
    def call(self):
        tensor=SimpleNamespace(is_cuda=True,shape=(1,3,224,224))
        for h in self.pre: h(self,(tensor,))
        for h in self.post: h(self,(tensor,),None)


@pytest.mark.parametrize('fail',[False,True])
def test_run_lifecycle_token_only(scene,tmp_path,monkeypatch,fail):
    plan,source=scene; models={k:TokenModel() for k in m.EXPECTED}
    receipts=plan['public_manifest']['model_spec']['loader_manifest']['expected_model_receipts']
    v4=SimpleNamespace(model_receipt=lambda model:{'RAM_fixture':True},native_root=lambda root:root)
    v5=SimpleNamespace(import_v4=lambda root:v4)
    v6=SimpleNamespace(official_root=lambda root:root)
    v7=SimpleNamespace(dependencies=lambda root:(None,v5,v6),imported_sources=lambda *args:{'fixture':str(source)})
    loader=SimpleNamespace(__file__=str(Path(m.base.owner.__file__).parent/(m.base.owner.LOADER+'.py')),
        helper=lambda root:v7,load_cuda=lambda *args:(models,{k:dict(exact_match=True) for k in ('predictor','proprio_encoder')},
            {'fixture':str(source)},receipts))
    monkeypatch.setattr(m.importlib,'import_module',lambda name:loader)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,mem_get_info=lambda:(8*1024**3,8*1024**3))))
    monkeypatch.delenv('PYTHONPATH',raising=False)
    def fake_goal(model,rgb):
        if fail:
            tensor=SimpleNamespace(is_cuda=True,shape=(1,3,224,224))
            for h in model.pre: h(model,(tensor,))
            raise RuntimeError('RAM injected forward failure')
        model.call(); return np.zeros(m.SHAPE,np.float32)
    monkeypatch.setattr(m,'cuda_goal',fake_goal)
    out=tmp_path/'results'/f'{m.TAG}_lifecycle'
    r=m.run_cuda(plan,out,main_authorized=True)
    assert all(v.removed==2 and not v.pre and not v.post for v in models.values())
    assert (out/'report.json').is_file()
    assert r['meter']['attempted']['official_encoder']==1
    if fail:
        assert r['status']=='failed' and r['meter']['completed']['official_encoder']==0
        assert len(r['words'])==4 and all(v['goal_patch_mse'] is None for v in r['words'])
        assert not (out/'goal_encoding.json').exists()
    else:
        assert r['status']=='all4_public_goal_scores_recorded'
        assert m.consume(plan,out/'report.json',m.sha(out/'report.json'))['selected_word']=='xy'


def test_native_material_size_regression():
    # True fixed dimensions, no neural arithmetic or real data scoring.
    assert np.zeros((256,384),np.float32).nbytes==393216
    assert 393216+1024+1024**2<m.CAP


def test_cli_guards_and_bundle(monkeypatch,tmp_path):
    assert (m.ROOT/'innerdocs'/(m.TAG+'.md')).is_file()
    with pytest.raises(ValueError,match='prediction authorities'):
        m.main(['prepare','--manifest',str(tmp_path/'new')])
    with pytest.raises(ValueError,match='actual scoring manifest'):
        m.main(['cuda','--manifest',str(tmp_path/'missing')])
    monkeypatch.setattr(m,'read_json',lambda *args:{})
    with pytest.raises(ValueError,match='explicit CUDA confirmation'):
        m.main(['cuda','--manifest',str(tmp_path/'missing'),'--manifest-sha256','0'*64])
