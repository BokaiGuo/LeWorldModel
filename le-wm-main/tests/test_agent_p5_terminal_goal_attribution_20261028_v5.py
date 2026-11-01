"""Real nine-file saved join plus non-neural hook/token lifecycle regressions."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v5 as m

ROOT=m.ROOT
REAL=tuple(ROOT/'results'/x for x in (
 'agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json',
 'agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json',
 'agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json',
 'agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json',
 'agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json',
 'agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json',
 'agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json',
 'agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json',
 'agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json'))

@pytest.fixture(scope='module')
def actual():
    # No monkeypatch/fake path/hash: actual pure-saved strict replay of all three.
    assert all(x.is_file() for x in REAL)
    value=m.prepare(*REAL)
    m.verify(value)
    return value

def test_actual_nine_files_prepare_verify_and_sha_closure(actual):
    assert actual['schema']==m.TAG
    q=actual['authority']['paths']
    assert [q[k] for k in ('goal_prepare','goal_report','goal_receipt','phys_manifest','phys_report','phys_receipt','num_manifest','num_report','num_receipt')]==[str(x.resolve()) for x in REAL]
    for p,h in actual['bindings'].items():
        assert Path(p).is_absolute() and len(h)==64 and m.impl.sha(p)==h
    assert actual['bindings'][actual['authority']['goalrec']['path']]==actual['authority']['goalrec']['sha256']

def test_real_cli_prepare_verify_roundtrip(actual,tmp_path):
    out=tmp_path/'manifest.json'
    assert m.main(['prepare','--manifest',str(out),'--authority',*[str(x) for x in REAL]])==0
    assert m.main(['verify','--manifest',str(out),'--manifest-sha256',m.impl.sha(out)])==0
    assert m.impl.read_json(out)==actual

@pytest.mark.parametrize('swap',[(3,6),(1,4),(4,7)])
def test_real_wrong_triples_rejected(swap):
    v=list(REAL); i,j=swap; v[i],v[j]=v[j],v[i]
    with pytest.raises(ValueError,match='authority SHA'): m.prepare(*v)

def test_actual_manifest_native_or_binding_tamper(actual):
    v=deepcopy(actual); v['authority']['terminals']['xy']['native']['state'][0]+=1
    with pytest.raises(ValueError,match='recomputation'): m.verify(v)
    v=deepcopy(actual); v['bindings'][str(REAL[0])]='0'*64
    with pytest.raises(ValueError,match='binding drift'): m.verify(v)

def test_no_global_protocol_recursion(actual):
    before=(m.impl.TAG,m.impl.protocol,m.impl.verify,m.impl.prepare)
    assert m.protocol()==actual['protocol']
    with pytest.raises(ValueError,match='confirmation'): m.run_cuda(actual,None,False)
    assert before==(m.impl.TAG,m.impl.protocol,m.impl.verify,m.impl.prepare)

def test_older_sources_unchanged():
    assert m.impl.sha(m.impl.__file__)==m.IMPL_SHA
    assert m.impl.sha(ROOT/'experiments/spectral_reachability/agent_p5_terminal_goal_attribution_20261028_v3.py')=='aa252fc8de5b103ed62ce4f08494c866ff6a18b5ca12c3770b74f71ee2d90d72'

class TokenModule:
    """Hook recorder only: not a neural model or physics simulator."""
    def __init__(self): self.pre=[]; self.post=[]; self.removed=0
    def handle(self,bank,h):
        def remove(): bank.remove(h); self.removed+=1
        return SimpleNamespace(remove=remove)
    def register_forward_pre_hook(self,h): self.pre.append(h); return self.handle(self.pre,h)
    def register_forward_hook(self,h): self.post.append(h); return self.handle(self.post,h)
    def call(self,fail=False):
        args=(SimpleNamespace(is_cuda=True,shape=(1,3,224,224)),)
        for h in self.pre: h(self,args)
        if fail: raise RuntimeError('RAM failure after attempted forward')
        for h in self.post: h(self,args,None)

def npy(path,a):
    with path.open('xb') as f: np.save(f,a,allow_pickle=False)
    return dict(path=str(path),sha256=m.impl.sha(path),bytes=path.stat().st_size,shape=list(a.shape),dtype=str(a.dtype))

@pytest.fixture
def scene(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); monkeypatch.setattr(m,'SHAPE',(2,3)); monkeypatch.setattr(m,'FLOOR',0)
    monkeypatch.setattr(m,'verify',lambda value:None); monkeypatch.delenv('PYTHONPATH',raising=False)
    (tmp_path/'results').mkdir()
    stub=tmp_path/'source.py'; stub.write_text('# RAM inert source\n')
    receipt={k:dict(RAM_only=True) for k in m.EXPECTED}
    a=dict(model_spec=dict(loader_manifest=dict(seals={str(stub):m.impl.sha(stub)},expected_model_receipts=receipt)),
        goalrec=npy(tmp_path/'goal.npy',np.zeros(m.SHAPE,np.float32)),prediction={},terminals={},
        gr={'new_meter':{'RAM':True}},nr={'meter':{'RAM':True}})
    for i,w in enumerate(m.WORDS):
        a['prediction'][w]=npy(tmp_path/(w+'_pred.npy'),np.full((3,*m.SHAPE),i,np.float32))
        a['terminals'][w]=dict(rgb=npy(tmp_path/(w+'_rgb.npy'),np.zeros((224,224,3),np.uint8)),
          native=dict(state=[1.]*7,proprio=[2.]*4,public_double=dict(bodies={'agent':{'position':[1.,2.]},'block':{'position':[3.,4.]}},env={'coverage_arr':[.25]}),transition=dict(done=False,reward=.25,info={'final_coverage':.25})))
    return dict(authority=a),stub,receipt

def setup_tokens(scene,monkeypatch,fail=False,drift=False,forbidden=False):
    value,stub,receipts=scene; models={k:TokenModule() for k in m.EXPECTED}
    def load(value):
        def after(): return {} if drift else receipts
        return models,{k:{'exact_match':True} for k in ('predictor','proprio_encoder')},{'RAM':str(stub)},receipts,after,lambda:{'RAM':str(stub)}
    monkeypatch.setattr(m,'load_runtime',load)
    def encode(model,rgb):
        if forbidden: models['predictor'].call()
        model.call(fail=fail); return np.ones(m.SHAPE,np.float32)
    monkeypatch.setattr(m,'encode',encode)
    return models

def test_real_runner_control_flow_no_monkeypatch_of_old_globals(scene,tmp_path,monkeypatch):
    value,_,_=scene; models=setup_tokens(scene,monkeypatch)
    before=(m.impl.TAG,m.impl.protocol,m.impl.verify)
    out=tmp_path/'results'/(m.TAG+'_RAM')
    r=m.run_cuda(value,out,True)
    assert r['status']=='all4_terminal_rgb_attribution_recorded' and r['errors']==r['finalization_errors']==[]
    assert len(r['words'])==4 and all(x.removed==2 and not x.pre and not x.post for x in models.values())
    assert before==(m.impl.TAG,m.impl.protocol,m.impl.verify)
    got=m.consume(value,out/'report.json',m.impl.sha(out/'report.json'))
    assert got['status']=='all4_saved_terminal_rgb_attribution_verified' and got['C2_established'] is False

@pytest.mark.parametrize('kind',['forward','model','forbidden'])
def test_runner_failure_no_refund_full_denominator_cleanup(scene,tmp_path,monkeypatch,kind):
    value,_,_=scene; models=setup_tokens(scene,monkeypatch,fail=kind=='forward',drift=kind=='model',forbidden=kind=='forbidden')
    out=tmp_path/'results'/(m.TAG+'_failure'); r=m.run_cuda(value,out,True)
    assert r['status']=='failed' and len(r['words'])==4 and all(x.removed==2 for x in models.values())
    if kind=='forward': assert r['meter']['attempted']['official_encoder']==1 and r['meter']['completed']['official_encoder']==0
    with pytest.raises(ValueError): m.consume(value,out/'report.json',m.impl.sha(out/'report.json'))

@pytest.mark.parametrize('kind',['token_bytes','token_sha','token_path','score','words','model','event','meter','extra','wall','science','source'])
def test_saved_mutations(scene,tmp_path,monkeypatch,kind):
    value,stub,_=scene; setup_tokens(scene,monkeypatch); out=tmp_path/'results'/(m.TAG+'_tamper')
    r=m.run_cuda(value,out,True); p=out/'report.json'
    if kind=='token_bytes': r['terminal_features']['xy']['bytes']+=1
    elif kind=='token_sha': r['terminal_features']['xy']['sha256']='0'*64
    elif kind=='token_path': r['terminal_features']['xy']['path']=r['terminal_features']['yx']['path']
    elif kind=='score': r['words'][0]['predicted_goal_mse']+=1
    elif kind=='words': r['words'].pop()
    elif kind=='model': r['model_after']={}
    elif kind=='event':
        e=m.impl.read_json(out/'event_0000.json'); e['value']={}; (out/'event_0000.json').write_text(json.dumps(e))
    elif kind=='meter': r['meter']['attempted']['official_encoder']=3
    elif kind=='extra': (out/'extra').write_text('preserved negative fixture')
    elif kind=='wall': r['wall_seconds']=m.WALL
    elif kind=='science': r['C2_established']=True
    else: stub.write_text('# drift\n')
    p.write_text(json.dumps(r))
    with pytest.raises((ValueError,KeyError)): m.consume(value,p,m.impl.sha(p))

def test_meter_caps_and_failure_no_refund():
    x=m.Meter(lambda *args:None); x.before('official_encoder',1)
    with pytest.raises(ValueError): x.before('official_encoder',1)
    assert x.attempted['official_encoder']==1 and x.completed['official_encoder']==0
    x.after('official_encoder',1)
    with pytest.raises(ValueError): x.before('predictor',1)
    with pytest.raises(ValueError): x.before('official_encoder',4)

def test_shape_regression(): assert np.zeros((256,384),np.float32).nbytes==393216
