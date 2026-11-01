import base64
import copy
import importlib.util
import subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('temporal_binding',ROOT/'experiments/spectral_reachability/agent_p1_external_temporal_binding_20261029_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def run(args):subprocess.run(args,check=True,capture_output=True)

@pytest.fixture
def signed(tmp_path):
    pubs=[];keys=[]
    for name in ('witness','authorizer'):
        key=tmp_path/(name+'.key');pub=tmp_path/(name+'.pem')
        run(['openssl','genpkey','-algorithm','ED25519','-out',str(key)])
        run(['openssl','pkey','-in',str(key),'-pubout','-out',str(pub)])
        pubs.append(pub.read_text());keys.append(key)
    policy=dict(schema=m.NAME+'.external_trust_policy',witness_public_key_pem=pubs[0],authorizer_public_key_pem=pubs[1])
    policysha=m.digest(m.canonical(policy));ss='a'*64;ps='b'*64;es='c'*64
    c=dict(schema=m.NAME+'.commitment',run_id='future-run-001',spec_sha256=ss,prepare_sha256=ps,executor_source_sha256=es,policy_sha256=policysha)
    a=dict(schema=m.NAME+'.launch_authorization',run_id=c['run_id'],commitment_sha256=m.digest(m.canonical(c)),prepare_sha256=ps,executor_source_sha256=es,policy_sha256=policysha,purpose='pre_dispatch_review_only')
    def sign(key,payload,n):
        src=tmp_path/(n+'.json');dst=tmp_path/(n+'.sig');src.write_bytes(m.canonical(payload))
        run(['openssl','pkeyutl','-sign','-inkey',str(key),'-rawin','-in',str(src),'-out',str(dst)])
        return dict(payload=payload,signature=base64.b64encode(dst.read_bytes()).decode())
    return policy,policysha,ss,ps,es,sign(keys[0],c,'c'),sign(keys[1],a,'a')

def test_valid_signatures_are_conditional_not_proof(signed):
    x=m.verify(*signed)
    assert x['status']=='signatures_valid_conditional_external_trust'
    assert not x['trusted_timestamp_proven'] and not x['dispatch_observed'] and not x['execution_authorized'] and not x['outcome_blinding_proven']

@pytest.mark.parametrize('kind',['spec','prepare','executor','signature','authorization_link','run_id','purpose','role_key','same_key','extra'])
def test_fail_closed_tampering(signed,kind):
    p,ph,ss,ps,es,c,a=copy.deepcopy(signed)
    if kind=='spec':ss='0'*64
    if kind=='prepare':ps='0'*64
    if kind=='executor':es='0'*64
    if kind=='signature':c['signature']=base64.b64encode(b'\0'*64).decode()
    if kind=='authorization_link':a['payload']['commitment_sha256']='0'*64
    if kind=='run_id':a['payload']['run_id']='other'
    if kind=='purpose':a['payload']['purpose']='dispatch'
    if kind=='role_key':p['witness_public_key_pem'],p['authorizer_public_key_pem']=p['authorizer_public_key_pem'],p['witness_public_key_pem']
    if kind=='same_key':p['authorizer_public_key_pem']=p['witness_public_key_pem']
    if kind=='extra':c['payload']['unknown']=True
    with pytest.raises(ValueError):m.verify(p,ph,ss,ps,es,c,a)

def test_duplicate_json_rejected():
    with pytest.raises(ValueError):m.parse(b'{"a":1,"a":2}')

def test_input_hash_and_symlink(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'abc');link=tmp_path/'link';link.symlink_to(p)
    assert m.read(p,m.digest(b'abc'))==b'abc'
    with pytest.raises(ValueError):m.read(p,'0'*64)
    with pytest.raises(ValueError):m.read(link,m.digest(b'abc'))

def test_tempdisk_roundtrip(signed,tmp_path):
    p=tmp_path/'result.json';result=m.verify(*signed);p.write_bytes(m.canonical(result))
    assert m.parse(p.read_bytes())==result

def test_cli_complete_disk_roundtrip(tmp_path):
    keys=[];pubs=[]
    for name in ('external_witness_fixture','external_authorizer_fixture'):
        key=tmp_path/(name+'.key');pub=tmp_path/(name+'.pem')
        run(['openssl','genpkey','-algorithm','ED25519','-out',str(key)])
        run(['openssl','pkey','-in',str(key),'-pubout','-out',str(pub)])
        keys.append(key);pubs.append(pub.read_text())
    files={}
    def put(name,data):
        path=tmp_path/name;path.write_bytes(data);files[name]=(path,m.digest(data))
    put('policy',m.canonical(dict(schema=m.NAME+'.external_trust_policy',witness_public_key_pem=pubs[0],authorizer_public_key_pem=pubs[1])))
    for name in ('spec','prepare','executor'):put(name,('fixture '+name).encode())
    c=dict(schema=m.NAME+'.commitment',run_id='fixture',spec_sha256=files['spec'][1],prepare_sha256=files['prepare'][1],executor_source_sha256=files['executor'][1],policy_sha256=files['policy'][1])
    a=dict(schema=m.NAME+'.launch_authorization',run_id='fixture',commitment_sha256=m.digest(m.canonical(c)),prepare_sha256=files['prepare'][1],executor_source_sha256=files['executor'][1],policy_sha256=files['policy'][1],purpose='pre_dispatch_review_only')
    for index,(name,obj) in enumerate((('commitment',c),('authorization',a))):
        src=tmp_path/(name+'.payload');sig=tmp_path/(name+'.sig');src.write_bytes(m.canonical(obj))
        run(['openssl','pkeyutl','-sign','-inkey',str(keys[index]),'-rawin','-in',str(src),'-out',str(sig)])
        put(name,m.canonical(dict(payload=obj,signature=base64.b64encode(sig.read_bytes()).decode())))
    output=tmp_path/(m.NAME+'_result.json');args=[]
    for name,(path,h) in files.items():args+=['--'+name,str(path),'--'+name+'-sha',h]
    script=str(ROOT/'experiments/spectral_reachability'/f'{m.NAME}.py')
    run([str(ROOT/'.venv/bin/python'),script,'check',*args,'--output',str(output)])
    run([str(ROOT/'.venv/bin/python'),script,'verify',*args,'--output',str(output)])
    assert m.parse(output.read_bytes())['input_files']['commitment']['sha256']==files['commitment'][1]
