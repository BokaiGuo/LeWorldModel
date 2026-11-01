import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('goal_semantics',ROOT/'experiments/spectral_reachability/agent_p1_prospective_goal_semantics_20261029_v1.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

@pytest.fixture
def sample(tmp_path):
    auth=tmp_path/'authority'; auth.write_bytes(b'fixture authority')
    png=tmp_path/'raw.rgb'; png.write_bytes(bytes(224*224*3))
    authority={'path':str(auth),'sha256':m.sha(auth.read_bytes())}
    im={'path':str(png),'sha256':m.sha(png.read_bytes()),'encoding':'RGB224_raw_u8'}
    goals=[[1.,2.,3.,4.,.5,4.,-3.],[1.,2.,3.,4.,.5,0.,0.]]
    slots=[dict(slot_id=str(i),root_id=str(i),goal7=g,goal_authority_sha256='a'*64) for i,g in enumerate(goals)]
    s=dict(schema=m.NAME+'.spec',slots=slots,checkpoints=[0,5],rgb_pairs=[['0','1']],authority=authority)
    rows=[]
    for slot in slots:
        st=[1.,2.,3.,4.,.5,0.,0.]; met=m.metrics(slot['goal7'],st)
        rows.append(dict(slot_id=slot['slot_id'],root_id=slot['root_id'],goal7=slot['goal7'][:],goal_authority_sha256=slot['goal_authority_sha256'],goal_rgb=copy.deepcopy(im),points=[dict(checkpoint=0,state7=st,native_success=met['native_success'],native_distance=met['native7_distance'],missing_reason=None),dict(checkpoint=5,state7=None,native_success=None,native_distance=None,missing_reason='native_early_stop')]))
    return s,dict(schema=m.NAME+'.receipt',authority=authority,rows=rows)

def test_full_saved_join_and_nonzero_velocity(sample):
    spec,receipt=sample; result=m.analyze(m.prepare(spec),receipt)
    assert result['denominator']==dict(slots=2,checkpoints_per_slot=2,rgb_pairs=1)
    p=result['slots'][0]['points'][0]
    assert p['metrics']['velocity_squared']==25 and p['metrics']['native_success'] is True
    assert p['metrics']['native7_squared']==25 and p['returned_distance']==5
    assert result['slots'][0]['points'][1]['metrics'] is None
    assert result['rgb_pairs'][0]['observed_rgb_collision'] is True

def test_image_missing_unknown(sample):
    s,r=sample; r['rows'][0]['goal_rgb']=None
    assert m.analyze(m.prepare(s),r)['rgb_pairs'][0]['observed_rgb_collision'] is None

def test_image_equal_without_velocity_difference_not_collision(sample):
    s,r=sample; s['slots'][1]['goal7']=s['slots'][0]['goal7'][:]; r['rows'][1]['goal7']=s['slots'][1]['goal7'][:]
    r['rows'][1]['points'][0]['native_distance']=5.
    assert not m.analyze(m.prepare(s),r)['rgb_pairs'][0]['observed_rgb_collision']

@pytest.mark.parametrize('kind',['goal','success','missing','checkpoint','identity','image','roster'])
def test_tampering_fails(sample,kind):
    s,r=copy.deepcopy(sample)
    if kind=='goal': r['rows'][0]['goal7'][5]=0.
    if kind=='success': r['rows'][0]['points'][0]['native_success']=False
    if kind=='missing': r['rows'][0]['points'][1]['missing_reason']=None
    if kind=='checkpoint': r['rows'][0]['points'][1]['checkpoint']=10
    if kind=='identity': r['rows'][0]['root_id']='other'
    if kind=='image': r['rows'][0]['goal_rgb']['sha256']='0'*64
    if kind=='roster': r['rows'].pop()
    with pytest.raises(ValueError): m.analyze(m.prepare(s),r)

def test_signed_zero_goal_identity(sample):
    s,r=sample; r['rows'][1]['goal7'][5]=-0.
    with pytest.raises(ValueError,match='goal identity'): m.analyze(m.prepare(s),r)

def test_signed_zero_is_not_velocity_difference(sample):
    s,r=sample
    s['slots'][0]['goal7'][5:]=[-0.,0.]; r['rows'][0]['goal7'][5:]=[-0.,0.]
    r['rows'][0]['points'][0]['native_distance']=0.
    pair=m.analyze(m.prepare(s),r)['rgb_pairs'][0]
    assert pair['velocity_bits_differ'] and not pair['velocity_values_differ'] and not pair['observed_rgb_collision']

def test_no_spec_slot_selection_or_duplicate_pair(sample):
    s,_=sample; s['rgb_pairs'].append(['1','0'])
    with pytest.raises(ValueError): m.prepare(s)

def test_real_tempdisk_write_rebuild_and_exclusive(sample,tmp_path):
    s,r=sample; expected=m.encoded(m.analyze(m.prepare(s),r)); path=tmp_path/(m.NAME+'_test.json')
    m.write(path,expected); assert path.read_bytes()==m.encoded(m.analyze(m.prepare(s),r))
    with pytest.raises(FileExistsError): m.write(path,expected)

def test_raw_angle_native_and_original_success():
    x=m.metrics([0,0,0,0,0,0,0],[0,0,0,0,7,0,0])
    assert x['raw_angle_squared']==49 and x['native_angle_term']<0 and x['native_success'] is True
