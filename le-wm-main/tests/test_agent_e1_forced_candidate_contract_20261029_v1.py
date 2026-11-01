from copy import deepcopy
import json
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_e1_forced_candidate_contract_20261029_v1 as m


def test_real_saved_join_fixed_denominator_and_sources():
    b=m.build()
    assert [(r['root'],r['method']) for r in b['slots']] == [(r,a) for r in (0,1) for a in m.METHODS]
    assert [r['bank_winner_index'] for r in b['slots']]==[48,48,48,55,55,55]
    assert b['protocol']['model_replay_axis_total']==24
    assert not b['production_authorized'] and not b['physical_reachability_authenticated']


def test_swapped_extra_and_bank_mismatch_rejected():
    r=m.old.parse_canonical(m.old.read_bound(m.REPORT,m.REPORT_SHA,m.old.RESERVE))
    payload,_=m.old.read_store(m.REPORT.parent,r)
    authority,_=m.host.authorities(); saved=payload['result.z']
    changed=deepcopy(saved);changed['choices'][0]['extra'][0][0]=0.7
    with pytest.raises(ValueError):m.plans(authority,changed)
    changed=deepcopy(saved);changed['results'][1]['candidates'][71]['cost']+=1
    with pytest.raises(ValueError):m.plans(authority,changed)


def test_temporary_cli_whole_verify_and_bad_sha(tmp_path):
    out=tmp_path/'contract.json';mod='experiments.spectral_reachability.'+m.NAME
    def cli(*args):return subprocess.run([sys.executable,'-B','-m',mod,*args],capture_output=True,text=True)
    prepared=cli('prepare','--output',str(out));assert prepared.returncode==0,prepared.stderr
    sha=json.loads(prepared.stdout)['sha256']
    assert cli('verify','--manifest',str(out),'--manifest-sha',sha).returncode==0
    assert cli('verify','--manifest',str(out),'--manifest-sha','0'*64).returncode!=0
    assert cli('prepare','--output',str(out)).returncode!=0
