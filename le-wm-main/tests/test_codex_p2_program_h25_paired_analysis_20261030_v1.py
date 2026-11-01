from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p2_program_h25_paired_analysis_20261030_v1 as a


@pytest.fixture(scope='module')
def saved():
    join=a.checked(a.JOIN,a.JOIN_SHA)
    report=a.checked(a.CUDA/'report.json',a.CUDA_SHA)
    branches=[a.checked(a.CUDA/r['file'],r['sha256']) for r in report['rows']]
    return join,report,branches


def test_complete_pairing(saved):
    out=a.analyze(*saved)
    assert len(out['pairs'])==4
    assert out['pairs'][0]['position_preference']=='tie'
    assert out['pairs'][2]['position_preference']=='library_chain'
    assert out['pairs'][2]['angle_preference']=='joint'
    assert all(not p['arms'][0]['native_success'] and not p['arms'][1]['native_success'] for p in out['pairs'])


@pytest.mark.parametrize('kind',['winner','action','context','stop','missing'])
def test_reject_changed_evidence(saved,kind):
    join,report,branches=deepcopy(saved)
    if kind=='winner': branches[0]['result']['winner_index']=-1
    elif kind=='action': branches[0]['result']['winner']['candidate']['raw_float32_hex']='00'*80
    elif kind=='context': join['policy_slots'][1]['row']['responses'][14]['state'][0]+=1
    elif kind=='stop': join['policy_slots'][0]['row']['terminated']=True
    else: join['policy_slots'].pop()
    with pytest.raises(ValueError):
        a.analyze(join,report,branches)


def test_hash_fails_closed(tmp_path):
    path=tmp_path/'bad.json';path.write_text('{}')
    with pytest.raises(ValueError):
        a.checked(path,'0'*64)
