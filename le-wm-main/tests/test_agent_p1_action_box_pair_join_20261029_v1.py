import copy
import importlib.util
import json
import struct
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('action_box_join', ROOT / 'experiments/spectral_reachability/agent_p1_action_box_pair_join_20261029_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.fixture(scope='module')
def inputs():
    return [json.loads((ROOT / m.INPUTS[k][0]).read_bytes()) for k in ('selector','evaluation','margin')]

def test_real_join_fixed_denominators(inputs):
    slots, pairs = m.join(*inputs)
    assert len(slots) == 40 and len(pairs) == 80
    assert sum(bool(p['missing_indices']) for p in pairs) == 4
    assert all(p['left']['outside_count'] == p['right']['outside_count'] == 0 for p in pairs)
    assert [(p['i'],p['j']) for p in pairs] == [(5*a+i,5*a+j) for a in range(8) for i in range(5) for j in range(i+1,5)]

def test_edges_signedzero_and_one_ulp():
    vals = [-1.,1.,-0.,0.,struct.unpack('<f',bytes.fromhex('ffff7f3f'))[0],1.25]+[0.]*14
    rows = m.distances(struct.pack('<20f',*vals))
    assert [r['exact_boundary'] for r in rows[:2]] == ['-1','+1']
    assert rows[2]['raw_bits_le'] == '00000080'
    assert rows[4]['nearest'] == 2**-24 and rows[4]['exact_boundary'] is None
    assert rows[5]['outside'] and rows[5]['nearest'] == -.25

def test_not_float32():
    with pytest.raises(ValueError, match='exactly float32'):
        m.raw([[0.1,0.]]*5,5)

@pytest.mark.parametrize('mutation', ['bits','identity','witness','pair','out_of_box'])
def test_join_rejects_tampering(inputs, mutation):
    a,b,c = copy.deepcopy(inputs)
    if mutation == 'bits': a['candidates'][0]['full10_raw_bits'] = 'ff'*80
    if mutation == 'identity': b['rows'][0]['task_id'] = 'other'
    if mutation == 'witness': b['exact_reuse_witnesses'][0]['actual20_verified'] = False
    if mutation == 'pair': c['pairs'][0]['j'] = 2
    if mutation == 'out_of_box':
        p=a['candidates'][0]['original_plan']; p['commands'][0][0]=2.; p['prefix_commands'][0][0]=2.
        bits=m.raw(p['commands'],10); pre=m.raw(p['prefix_commands'],5)
        a['candidates'][0]['full10_raw_bits']=bits.hex()
        for obj in (b['exact_reuse_witnesses'][0],c['slots'][0]):
            obj['full10_sha256']=m.sha(bits); obj['prefix_sha256']=m.sha(pre)
    with pytest.raises(ValueError): m.join(a,b,c)

def test_entire_report_serialization_and_detect_changes(tmp_path, inputs):
    slots,pairs=m.join(*inputs)
    value={'slots':slots,'pairs':pairs}
    path=tmp_path/'report.json'; path.write_bytes(m.encoded(value))
    assert path.read_bytes()==m.encoded(value)
    value['pairs'][0]['j']=4
    assert path.read_bytes()!=m.encoded(value)

def test_pinned_sources_and_build():
    assert m.build()['counts']['pairs'] == 80
