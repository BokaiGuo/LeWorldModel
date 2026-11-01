from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
from experiments.spectral_reachability import agent_p2_intersection_rgb_pair_diagnostic_20261029_v1 as m

ROOT=Path(__file__).resolve().parents[1]/'results'
RGB=ROOT/'agent_p2_intersection_rgb_20261029_v1_consume_MAIN01.json'
REPORT=ROOT/'agent_p2_intersection_rgb_20261029_v1_cuda_MAIN01/report.json'
PHYS=ROOT/'agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json'


@pytest.fixture(scope='module')
def data():
    return json.loads(RGB.read_bytes()),json.loads(REPORT.read_bytes()),json.loads(PHYS.read_bytes())


def analyze(data):
    rgb,report,physical=data
    return m.analyze(rgb,report,physical,lambda d:m.load_latent(REPORT.parent,d,report['artifact_sha256']))


def test_real_saved_fixed_denominator(data):
    r=analyze(data)
    assert len(r['pairs'])==18 and r['denominator']==dict(slots=6,positions=24,pairs=18,rgb_comparable=16,native_comparable=16)
    absent=[x for x in r['pairs'] if not x['comparable_rgb']]
    assert len(absent)==2 and all(x['root']==1 and x['checkpoint']==15 and x['left']=='joint' for x in absent)
    assert all(x['actual_delta_b_minus_a'] is None and x['native_delta_b_minus_a'] is None and x['actual_sign'] is None for x in absent)
    assert all(x['native'] is None for x in r['rows'] if x['checkpoint']==0)


def test_actual_build_pinned_authority():
    assert m.build(RGB,REPORT,PHYS)['denominator']['pairs']==18


def test_q_and_cross_arithmetic():
    assert m.q([2.0]*192,[1.0]*192)==1
    assert m.sign(0)==0 and m.sign(None) is None and m.sign(-1)==-1
    n=m.native([1,2,3,4,5,6,7],[0]*7,dict(reward=-1,native_success=False))
    assert n['axis_squared']==[1,4,9,16,25,36,49] and n['agent_velocity_squared']==85 and n['native7_squared']==140


@pytest.mark.parametrize('field,value', [('raw15_bits','00'),('checkpoint',10),('model_Q',999),('prediction_role','forecast')])
def test_receipt_tamper_rejected(data,field,value):
    r,p,x=deepcopy(data);r['rows'][1][field]=value
    with pytest.raises(ValueError):analyze((r,p,x))


def test_artifact_bytes_tamper_rejected(data):
    r,p,x=deepcopy(data);d=r['rows'][0]['latent'];p['artifact_sha256'][d['file']]='0'*64
    with pytest.raises(ValueError,match='inventory'):analyze((r,p,x))


def test_missing_frame_does_not_erase_native_return(data):
    r,p,x=deepcopy(data);index=1
    for row in (r['rows'][index],p['rows'][index]):
        row.update(frame=None,latent=None,missing_reason='physical_checkpoint_missing')
    r['rows'][index].update(actual_Q=None,model_actual_MSE=None)
    x['rows'][0]['checkpoints']=[c for c in x['rows'][0]['checkpoints'] if c['step']!=5]
    out=analyze((r,p,x))
    assert out['rows'][index]['actual_Q'] is None and out['rows'][index]['native'] is not None
    pair=next(t for t in out['pairs'] if t['root']==0 and t['checkpoint']==5 and t['left']=='joint' and t['right']=='composition')
    assert not pair['comparable_rgb'] and pair['comparable_native']


def test_whole_tempfile_analyze_verify(tmp_path,monkeypatch):
    out=tmp_path/(m.NAME+'_fixture.json')
    args=['tool','analyze','--rgb-receipt',str(RGB),'--rgb-report',str(REPORT),'--physical-receipt',str(PHYS),'--output',str(out)]
    monkeypatch.setattr(sys,'argv',args);m.main();h=m.sha(out.read_bytes())
    monkeypatch.setattr(sys,'argv',[*args[:1],'verify',*args[2:],'--output-sha',h]);m.main()
    out.write_bytes(out.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA'):m.main()
