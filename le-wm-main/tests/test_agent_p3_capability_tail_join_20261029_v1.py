from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
from experiments.spectral_reachability import agent_p3_capability_tail_join_20261029_v1 as m

ROOT=Path(__file__).resolve().parents[1]/'results'
THREE=ROOT/'agent_p3_capability_three_layer_summary_20261029_v1_MAIN01.json'
TAIL=ROOT/'agent_p3_settling_tail_summary_20261029_v1_MAIN01.json'


@pytest.fixture(scope='module')
def inputs():return json.loads(THREE.read_bytes()),json.loads(TAIL.read_bytes())


def test_actual_roster_and_retained_controls(inputs):
    r=m.analyze(*inputs)
    assert r['denominator']==dict(branches=10,endpoints=5,pair_times=10)
    assert len(r['rows'])==10 and len(r['pairs'])==5
    assert all(x['cp20_model_Q'] is None and x['cp20_real_Q'] is None for x in r['rows'])
    assert r['C_control_pairs']==inputs[0]['original_C_control_pairs'] and not r['P3_admission']


@pytest.mark.parametrize('field',['state','pixel_sha256','file_sha256'])
def test_cp15_identity_tamper_rejected(inputs,field):
    t,b=deepcopy(inputs);t['branches'][0]['source'][field]='bad'
    with pytest.raises(ValueError):m.analyze(t,b)


def test_same_context_mismatch_rejected(inputs):
    t,b=deepcopy(inputs);t['pairs'][0]['same_parent_context']=False
    with pytest.raises(ValueError,match='same-context'):m.analyze(t,b)


def test_missing_tail_retains_fixed_pair(inputs):
    t,b=deepcopy(inputs);b['branches'][0]['checkpoints']['20']=None
    result=m.analyze(t,b);assert len(result['pairs'])==5
    assert result['pairs'][0]['native_gain']['20'] is None and result['pairs'][0]['change_cp20_minus_cp15'] is None
    assert result['pairs'][0]['real_gain_cp15'] is not None


def test_missing_cp15_is_null_not_last_endpoint(inputs):
    t,b=deepcopy(inputs);v=t['branches'][0];v['source']=None;v['native_metrics']=None;v['actual_latent192']=None
    b['branches'][0]['checkpoints']['15']=None;t['pairs'][0]['real_latent_gain']['value']=None
    result=m.analyze(t,b);assert result['rows'][0]['cp15_native'] is None
    assert result['pairs'][0]['real_gain_cp15'] is None and result['pairs'][0]['native_gain']['15'] is None


def test_goal_velocity_and_sign():
    p=dict(metrics=dict(state7=[0,0,0,0,0,2,3],goal7=[0,0,0,0,0,1,1]),native_success=True,native_reward=-1,pixel_sha256='x',png_sha256='y')
    r=m.metrics(p,p['metrics']['goal7']);assert r['agent_velocity_squared']==5 and r['native7_squared']==5
    assert m.sign(0)=='tie' and m.sign(None) is None


def test_real_inputs_tempfile_build_verify(tmp_path,monkeypatch):
    out=tmp_path/(m.NAME+'_test.json');argv=['tool','build','--three',str(THREE),'--tail',str(TAIL),'--output',str(out)]
    monkeypatch.setattr(sys,'argv',argv);m.main();h=m.hashlib.sha256(out.read_bytes()).hexdigest()
    monkeypatch.setattr(sys,'argv',[argv[0],'verify',*argv[2:],'--output-sha',h]);m.main()
    out.write_bytes(out.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA'):m.main()
