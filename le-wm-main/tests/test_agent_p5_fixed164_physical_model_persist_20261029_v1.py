import copy
import hashlib
import pytest
from experiments.spectral_reachability import agent_p5_fixed164_physical_model_persist_20261029_v1 as m

@pytest.fixture
def fixture_analysis():
 return dict(schema=m.join.NAME+'.analysis',status='fixed164_saved_identity_join_only',rows=[{'index':i} for i in range(164)],pairs=[{'index':i} for i in range(80)],baseline_indices=[0,41,82,123],known_roots=2,fresh_roots=0,P1_admission=False,P5_admission=False,input_sha256={k:v[1] for k,v in m.join.FILES.items()})

@pytest.fixture
def fake_join(monkeypatch,fixture_analysis):
 monkeypatch.setattr(m.join,'load',lambda:{'fixture':True})
 monkeypatch.setattr(m.join,'analyze',lambda _:copy.deepcopy(fixture_analysis))
 return fixture_analysis

def test_build_fixed_boundaries(fake_join):
 v=m.build(); assert v['denominator']==dict(branches=164,pairs=80,baselines=4,known_roots=2,fresh_certified_roots=0)
 assert len(v['source_closure'])==6 and v['cross_model_spaces_comparable'] is False and not v['full_strict_rerun']

def test_exclusive_write_verify_and_mutations(tmp_path,fake_join):
 path=tmp_path/'result.json'; raw=m.join.canonical(m.build()); m.write(path,raw); sha=hashlib.sha256(raw).hexdigest()
 assert m.verify(path,sha)['status']=='exact_rebuild_verified'
 with pytest.raises(FileExistsError): m.write(path,b'other')
 assert path.read_bytes()==raw
 changed=raw.replace(b'"fresh_roots":0',b'"fresh_roots":1'); path.write_bytes(changed)
 with pytest.raises(ValueError): m.verify(path,sha)
 with pytest.raises(ValueError): m.verify(path,hashlib.sha256(changed).hexdigest())

@pytest.mark.parametrize('mutate',[
 lambda x:x['rows'].pop(), lambda x:x['pairs'].pop(), lambda x:x.update(fresh_roots=1),
 lambda x:x.update(P5_admission=True), lambda x:x['input_sha256'].update(physical='0'*64),
 lambda x:x.update(baseline_indices=[0,41,82,124]),
])
def test_fixed_contract_refuses_mutation(fake_join,mutate):
 mutate(fake_join)
 with pytest.raises(ValueError): m.build()

def test_failure_and_namespace(tmp_path,monkeypatch):
 p=tmp_path/'link'; p.symlink_to(tmp_path/'missing')
 with pytest.raises(FileExistsError): m.write(p,b'x')
 with pytest.raises(ValueError): m.write(tmp_path/'large',b'x'*(m.CAP+1))
 monkeypatch.setattr(m.shutil,'disk_usage',lambda _:type('D',(),{'free':0})())
 with pytest.raises(ValueError): m.write(tmp_path/'no-space',b'x')
 with pytest.raises(ValueError): m.target('/tmp/other.json')

def test_source_closure_tamper(monkeypatch):
 changed=dict(m.SEALED); changed[next(iter(changed))]='0'*64
 monkeypatch.setattr(m,'SEALED',changed)
 with pytest.raises(ValueError): m.closure()
