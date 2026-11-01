"""Non-neural simulator fixtures, not physics efficacy or CUDA evidence."""
import json
import numpy as np
import pytest
from experiments.spectral_reachability import p2_full20_physical_20261027_v2 as p
from tests.test_p2_matched_sampling_physical_20261027_v1 import Env, state, plan


@pytest.mark.parametrize('missing,steps,status', [(None,10,'completed'),(3,3,'observation_unavailable'),(0,0,'observation_unavailable')])
def test_arbitrary_perstep_actions_retained(tmp_path,missing,steps,status):
    value=plan()
    value['commands']=np.linspace(-.9,.9,20).reshape(10,2).tolist()
    row=p.physical_branch(plan=value,output=tmp_path,reset=lambda:{},env=Env(missing_at=missing),
        state_of=state,evaluate=lambda s:{'official':{},'pose':{}})
    assert row['status']==status and row['completed_commands']==steps
    assert [e['command'] for e in row['trace']]==value['commands'][:steps]


@pytest.mark.parametrize('missing,encoded',[(None,1),(3,0)])
def test_fixed_horizon_encoding_never_substitutes_guard_endpoint(tmp_path,missing,encoded):
    row=p.physical_branch(plan=plan(),output=tmp_path,reset=lambda:{},env=Env(missing_at=missing),
        state_of=state,evaluate=lambda s:{'official':{},'pose':{}})
    p.write(tmp_path/'physical_report.json',{'rows':[row]})
    calls=[]
    def encode(path):
        calls.append(path)
        return np.zeros(192).tolist()
    receipt=p.optional_encodings([row],tmp_path,lambda:encode)[0]
    assert len(calls)==receipt['attempted_encode_samples']==encoded
    assert receipt['status']==('completed' if encoded else 'blocked_fixed_horizon')
    assert len(list(tmp_path.glob('*.physical.json')))==1


def test_encoding_setup_failure_durable(tmp_path):
    rows=[]
    p.materialize_missing([plan()],rows,tmp_path)
    p.write(tmp_path/'physical_report.json',{'rows':rows})
    def fail():raise RuntimeError('fixture CUDA failure')
    receipt=p.optional_encodings(rows,tmp_path,fail)[0]
    assert receipt['status']=='encoding_setup_failed' and receipt['attempted_encode_samples']==0
    assert json.loads(next(tmp_path.glob('*.encoding.json')).read_text())==receipt


def test_report_must_precede_encoding(tmp_path):
    with pytest.raises(ValueError,match='persist'):
        p.optional_encodings([],tmp_path,lambda:pytest.fail('premature CUDA setup'))
