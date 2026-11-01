import copy
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('intersection_goal_v2',ROOT/'experiments/spectral_reachability/agent_p1_intersection_goal_adapter_20261029_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def fixture(tmp_path):
    h='a'*64;src=tmp_path/'source.py';src.write_bytes(b'fake source');sh=m.digest(src.read_bytes())
    roots=[dict(root_slot=i,root_sha256=m.ROOTS[i],goal=[float(i)]*7) for i in range(2)]
    plans=[dict(index=i,root_slot=r,condition=c,arm=a,native_root=copy.deepcopy(roots[r]),selection_uses_physics=False,selection_digest=h) for i,(r,c,a) in enumerate(m.ORDER)]
    contract=dict(schema=m.PHYSICAL+'.contract',fixed_denominator=6,order=[list(x) for x in m.ORDER],checkpoints=[0,5,10,15],horizon=15,condition='off',known_roots=2,fresh_roots=0,selection_uses_physics=False,P1_admission=False)
    authority=dict(producer='agent_p2_intersection_host_cuda_20261029_v2',planner_receipt_sha=m.RECEIPT_SHA,complete_consume_rebuilt=True,persisted_receipt_equal=True,selection_uses_physics=False,missing_exports=[])
    p=dict(schema=m.PHYSICAL+'.prepare',contract=contract,bindings={str(src):sh},planner_authority=authority,source_roots=roots,plans=plans)
    return p,tmp_path/'prepared.json',h,src,sh

def test_projection(fixture):
    p,path,h,src,sh=fixture;x=m.project(p,path,h,src,sh)
    assert len(x['slots'])==6 and x['checkpoints']==[0,5,10,15] and x['rgb_pairs']==[]
    assert [x['root_id'] for x in x['slots']]==[m.ROOTS[0]]*3+[m.ROOTS[1]]*3

@pytest.mark.parametrize('kind',['old_schema','old_contract','denominator','order','checkpoints','goal','source','authority','missing','root'])
def test_reject_incompatible(fixture,kind):
    p,path,h,src,sh=fixture;p=copy.deepcopy(p)
    if kind=='old_schema':p['schema']=m.PHYSICAL.replace('_v2','_v1')+'.prepare'
    if kind=='old_contract':p['contract']['schema']='old.contract'
    if kind=='denominator':p['contract']['fixed_denominator']=4
    if kind=='order':p['contract']['order'].reverse()
    if kind=='checkpoints':p['contract']['checkpoints']=[0,5,10]
    if kind=='goal':p['plans'][0]['native_root']['goal'][0]=2.
    if kind=='source':p['bindings'][str(src)]='b'*64
    if kind=='authority':p['planner_authority']['planner_receipt_sha']='b'*64
    if kind=='missing':p['plans'][0]=None
    if kind=='root':p['source_roots'][0]['root_sha256']='b'*64
    with pytest.raises((ValueError,KeyError)):m.project(p,path,h,src,sh)

def test_signed_zero_bits():
    assert m.goal_bits([0.]*7)!=m.goal_bits([-0.]+[0.]*6)

def test_temporary_roundtrip(fixture,tmp_path):
    p,path,h,src,sh=fixture;data=m.encode(m.project(p,path,h,src,sh));out=tmp_path/'fixture.json'
    with out.open('xb') as f:f.write(data)
    assert out.read_bytes()==m.encode(m.project(p,path,h,src,sh))
    with pytest.raises(FileExistsError):out.open('xb')
