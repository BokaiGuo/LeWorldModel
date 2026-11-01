import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('intersection_goal_adapter',ROOT/'experiments/spectral_reachability/agent_p1_intersection_goal_adapter_20261029_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture(scope='module')
def p(): return json.loads(m.read_bound(m.PREP,m.PREP_SHA))

def test_actual_abi_fixed_roster_and_goal(p):
    x=m.project(p,m.PREP,m.PREP_SHA)
    assert len(x['slots'])==6 and x['checkpoints']==[0,5,10,15] and x['rgb_pairs']==[]
    assert [s['root_id'] for s in x['slots']]==[m.ROOTS[0]]*3+[m.ROOTS[1]]*3
    assert x['slots'][0]['goal7']==p['source_roots'][0]['goal']
    assert x['authority']['sha256']==m.PREP_SHA

@pytest.mark.parametrize('change',['denominator','order','checkpoint','goal','signed_zero','selection','root','missing','producer'])
def test_refuses_schema_or_binding_drift(p,change):
    x=copy.deepcopy(p)
    if change=='denominator':x['contract']['fixed_denominator']=6
    if change=='order':x['contract']['order'].reverse()
    if change=='checkpoint':x['contract']['checkpoints']=[0,5,10]
    if change=='goal':x['plans'][0]['native_root']['goal'][0]+=1
    if change=='signed_zero':x['plans'][0]['native_root']['goal'][0]=-0.
    if change=='selection':x['plans'][0]['selection_uses_physics']=True
    if change=='root':x['source_roots'][0]['root_sha256']='0'*64
    if change=='missing':x['plans'][0]=None
    if change=='producer':x['planner_authority']['planner_receipt_sha']='0'*64
    with pytest.raises(ValueError):m.project(x,m.PREP,m.PREP_SHA)

def test_float_bits_distinguish_zero():
    assert m.goal_bits([0.]*7)!=m.goal_bits([-0.]+[0.]*6)

def test_actual_inputs_closure_and_temp_roundtrip(tmp_path):
    data=m.encoded(m.build());f=tmp_path/(m.NAME+'_fixture.json')
    with f.open('xb') as out:out.write(data)
    assert f.read_bytes()==m.encoded(m.build())
    with pytest.raises(FileExistsError):f.open('xb')
