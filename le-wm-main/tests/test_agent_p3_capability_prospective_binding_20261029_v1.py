import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from experiments.spectral_reachability import agent_p3_capability_prospective_binding_20261029_v1 as m


def put(root,name,data):
    p=root/name;p.write_bytes(data);return str(p),hashlib.sha256(data).hexdigest()


@pytest.fixture
def proposal(tmp_path):
    args=[]
    for name in ('cohort','prepare','executor'):args.extend(put(tmp_path,name,('fixture '+name).encode()))
    return m.proposal('fixture-run',*args)


def test_fixed_accounting_and_never_ready(proposal):
    c=proposal['contract']
    assert c['cp5_signals']+c['cp10_signals']==c['J']==7 and c['VJP']==7*192
    assert c['predict_API']==7*2+5*2 and c['Q_calls']==5+7+10
    assert c['stage_A_physics']['png']==264 and c['stage_C_physics']['png']==810
    assert proposal['execution_authorized'] is False and proposal['future_cohort_authority_verified'] is False


def test_hash_and_alias_fail_closed(tmp_path,proposal):
    c,p,e=(proposal[k] for k in ('future_cohort','future_prepare','future_executor'))
    with pytest.raises(ValueError,match='SHA mismatch'):m.proposal('fixture-run',c['path'],'0'*64,p['path'],p['sha256'],e['path'],e['sha256'])
    with pytest.raises(ValueError,match='separate'):m.proposal('fixture-run',c['path'],c['sha256'],c['path'],c['sha256'],e['path'],e['sha256'])
    link=tmp_path/'link';link.symlink_to(c['path'])
    with pytest.raises(ValueError,match='canonical'):m.byte_hash(link,c['sha256'])


def test_reject_existing_known_task_bundle(tmp_path,monkeypatch,proposal):
    real=m.byte_hash
    def pretend(path,sha,cap=128*1024**2):
        if sha==m.OLD_ENDPOINT_TASKS_SHA:return dict(path=str(path),sha256=sha)
        return real(path,sha,cap)
    monkeypatch.setattr(m,'byte_hash',pretend)
    c,p,e=(proposal[k] for k in ('future_cohort','future_prepare','future_executor'))
    with pytest.raises(ValueError,match='known observed'):m.proposal('fixture',c['path'],m.OLD_ENDPOINT_TASKS_SHA,p['path'],p['sha256'],e['path'],e['sha256'])


def test_altered_proposal_refused_before_signature(proposal):
    p=deepcopy(proposal);p['contract']['J']=8
    with pytest.raises(ValueError,match='contract'):m.signed_check(p,'a'*64,{},'b'*64,{}, {})


def test_temporary_signed_full_cli(tmp_path,proposal,monkeypatch):
    ext=m.external;keys=[];pubs=[]
    for name in ('witness','authorizer'):
        key=tmp_path/(name+'.key');pub=tmp_path/(name+'.pub')
        subprocess.run(['openssl','genpkey','-algorithm','ED25519','-out',str(key)],check=True,capture_output=True)
        subprocess.run(['openssl','pkey','-in',str(key),'-pubout','-out',str(pub)],check=True,capture_output=True)
        keys.append(key);pubs.append(pub.read_text())
    pp,ph=put(tmp_path,'proposal',m.encoded(proposal))
    policy=dict(schema=ext.NAME+'.external_trust_policy',witness_public_key_pem=pubs[0],authorizer_public_key_pem=pubs[1])
    policy_path,policy_sha=put(tmp_path,'policy',ext.canonical(policy))
    prepare_sha=proposal['future_prepare']['sha256'];executor_sha=proposal['future_executor']['sha256']
    c=dict(schema=ext.NAME+'.commitment',run_id=proposal['run_id'],spec_sha256=ph,prepare_sha256=prepare_sha,executor_source_sha256=executor_sha,policy_sha256=policy_sha)
    a=dict(schema=ext.NAME+'.launch_authorization',run_id=proposal['run_id'],commitment_sha256=ext.digest(ext.canonical(c)),prepare_sha256=prepare_sha,executor_source_sha256=executor_sha,policy_sha256=policy_sha,purpose='pre_dispatch_review_only')
    refs=[]
    for name,obj,key in zip(('commitment','authorization'),(c,a),keys,strict=True):
        raw=tmp_path/(name+'.payload');sig=tmp_path/(name+'.sig');raw.write_bytes(ext.canonical(obj))
        subprocess.run(['openssl','pkeyutl','-sign','-inkey',str(key),'-rawin','-in',str(raw),'-out',str(sig)],check=True,capture_output=True)
        path,h=put(tmp_path,name,ext.canonical(dict(payload=obj,signature=base64.b64encode(sig.read_bytes()).decode())))
        refs += ['--'+name,path,'--'+name+'-sha',h]
    output=tmp_path/(m.NAME+'_fixture.json')
    args=['prog','check','--proposal',pp,'--proposal-sha',ph,'--policy',policy_path,'--policy-sha',policy_sha,*refs,'--output',str(output)]
    monkeypatch.setattr(sys,'argv',args);m.main();result=json.loads(output.read_bytes())
    assert result['signature_check']['status']=='signatures_valid_conditional_external_trust' and result['execution_authorized'] is False
    h=hashlib.sha256(output.read_bytes()).hexdigest();monkeypatch.setattr(sys,'argv',[args[0],'verify',*args[2:],'--output-sha',h]);m.main()
    output.write_bytes(output.read_bytes()+b' ')
    with pytest.raises(ValueError,match='hash mismatch'):m.main()
