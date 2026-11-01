from copy import deepcopy
import subprocess
import sys
import numpy as np
import pytest

from experiments.spectral_reachability import p2_h15_sra_prepare_20261029 as m


@pytest.fixture
def inputs():
    return [m.read(m.PREPARE,m.PREPARE_SHA), m.read(m.RECEIPT,m.RECEIPT_SHA)]


def test_real_preparation(inputs):
    result=m.assemble(*inputs)
    assert len(result['cases'])==2 and result['design']['raw_steps']==15
    assert result['design']['candidate_count']==40 and not result['ready_to_execute']
    assert result['new_neural_calls']==result['new_physical_steps']==0
    assert all('raw15' not in x and 'logits' not in x for x in result['cases'])


@pytest.mark.parametrize('field', ['goal','root_context','warm_raw10','task_id'])
def test_no_cross_arm_mismatch(inputs,field):
    p,r=deepcopy(inputs)
    p['plans'][1][field]='mismatch'
    with pytest.raises((ValueError,KeyError)):
        m.assemble(p,r)


def test_no_cross_receipt_root_or_goal(inputs):
    p,r=deepcopy(inputs)
    r['rows'][4]['goal'][0]+=1
    with pytest.raises(ValueError,match='native root/goal'):
        m.assemble(p,r)


def test_early_stop_not_replaced_with_new_outcome(inputs):
    p,r=inputs
    assert r['rows'][5]['own_returned']==14
    result=m.assemble(p,r)
    assert result['cases'][1]['root_slot']==1
    assert result['ready_to_execute'] is False


def test_h15_bank_identity_and_reproducible_signs():
    j=np.zeros((192,30)); j[:30,:]=np.diag(np.arange(30,0,-1))
    a=m.spectral_bank(j,np.zeros(192),np.ones(192))
    b=m.spectral_bank(j,np.zeros(192),np.ones(192))
    assert a==b and len(a['labels'])==40
    assert np.asarray(a['raw15']).shape==(40,15,2)
    assert np.max(np.linalg.norm(a['logits'],axis=1)) <= m.CAP*(1+1e-12)
    assert a['labels'][0]=='zero' and len(set(a['labels']))==40
    assert np.allclose(a['raw15'],np.tanh(a['logits']).astype(np.float32).reshape(40,15,2))


def test_invalid_jacobian_rejected():
    with pytest.raises(ValueError,match='finite array'):
        m.spectral_bank(np.zeros((192,20)),np.zeros(192),np.zeros(192))


def test_bad_pinned_input(tmp_path):
    p=tmp_path/'x'; p.write_text('{}')
    with pytest.raises(ValueError,match='SHA'):
        m.read(p,'0'*64)


def test_real_task_sources_and_exclusive_output(tmp_path):
    built=m.build()
    assert all(len(c['action_normalizer']['std'])==2 for c in built['cases'])
    out=tmp_path/(m.NAME+'_test.json')
    command=[sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'prepare','--output',str(out)]
    first=subprocess.run(command,cwd=m.ROOT,capture_output=True)
    assert first.returncode==0, first.stderr
    original=out.read_bytes()
    second=subprocess.run(command,cwd=m.ROOT,capture_output=True)
    assert second.returncode!=0 and out.read_bytes()==original
