"""Verify externally signed pre-dispatch commitment; never launch or read outcomes."""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

NAME='agent_p1_external_temporal_binding_20261029_v1'
SHA=re.compile(r'^[0-9a-f]{64}$')
RUN=re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')

def need(value,message):
    if not value: raise ValueError(message)

def digest(value):return hashlib.sha256(value).hexdigest()

def read(path,expected,cap=128*1024*1024):
    need(isinstance(expected,str) and SHA.fullmatch(expected),'explicit SHA256')
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded absolute regular input')
    data=path.read_bytes();need(len(data)<=cap and digest(data)==expected,'input hash mismatch');return data

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')

def parse(data):
    def pairs(items):
        d={}
        for k,v in items:
            need(k not in d,'duplicate JSON key');d[k]=v
        return d
    return json.loads(data,object_pairs_hook=pairs)

def ed25519_key_id(key):
    try:r=subprocess.run(['openssl','pkey','-pubin','-outform','DER'],input=key,capture_output=True,timeout=10,check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:raise ValueError('external verifier unavailable') from exc
    der=r.stdout
    need(r.returncode==0 and len(der)==44 and der[:12]==bytes.fromhex('302a300506032b6570032100'),'Ed25519 public key required')
    return digest(der)

def verify_signature(key, payload, signature):
    try:sig=base64.b64decode(signature,validate=True)
    except (ValueError,TypeError) as exc:raise ValueError('signature base64') from exc
    need(len(sig)==64,'Ed25519 signature length')
    with tempfile.TemporaryDirectory(prefix='p1-external-binding-') as d:
        p=Path(d); (p/'key.pem').write_bytes(key);(p/'payload').write_bytes(canonical(payload));(p/'signature').write_bytes(sig)
        try:
            r=subprocess.run(['openssl','pkeyutl','-verify','-pubin','-inkey',str(p/'key.pem'),'-rawin','-in',str(p/'payload'),'-sigfile',str(p/'signature')],capture_output=True,timeout=10,check=False)
        except (OSError,subprocess.TimeoutExpired) as exc:raise ValueError('external verifier unavailable') from exc
        need(r.returncode==0,'invalid external signature')

def verify(policy, policy_sha, spec_sha, prepare_sha, executor_sha, commitment, authorization):
    need(set(policy)=={'schema','witness_public_key_pem','authorizer_public_key_pem'},'policy fields')
    need(policy['schema']==NAME+'.external_trust_policy','policy schema')
    try: witness=policy['witness_public_key_pem'].encode('ascii'); authorizer=policy['authorizer_public_key_pem'].encode('ascii')
    except (AttributeError,UnicodeEncodeError) as exc:raise ValueError('public key format') from exc
    need(ed25519_key_id(witness)!=ed25519_key_id(authorizer),'distinct role keys required')
    for val in (policy_sha,spec_sha,prepare_sha,executor_sha):need(isinstance(val,str) and SHA.fullmatch(val),'SHA fields')
    need(set(commitment)=={'payload','signature'} and set(authorization)=={'payload','signature'},'signed envelopes')
    c=commitment['payload'];a=authorization['payload']
    need(set(c)=={'schema','run_id','spec_sha256','prepare_sha256','executor_source_sha256','policy_sha256'} and c['schema']==NAME+'.commitment','commitment fields')
    need(isinstance(c['run_id'],str) and RUN.fullmatch(c['run_id']),'unique run ID')
    need((c['spec_sha256'],c['prepare_sha256'],c['executor_source_sha256'],c['policy_sha256'])==(spec_sha,prepare_sha,executor_sha,policy_sha),'exact commit binding')
    verify_signature(witness,c,commitment['signature'])
    need(set(a)=={'schema','run_id','commitment_sha256','prepare_sha256','executor_source_sha256','policy_sha256','purpose'} and a['schema']==NAME+'.launch_authorization','authorization fields')
    need(a['run_id']==c['run_id'] and a['commitment_sha256']==digest(canonical(c)) and a['prepare_sha256']==prepare_sha and a['executor_source_sha256']==executor_sha and a['policy_sha256']==policy_sha and a['purpose']=='pre_dispatch_review_only','authorization binds commitment')
    verify_signature(authorizer,a,authorization['signature'])
    return dict(schema=NAME+'.preflight',status='signatures_valid_conditional_external_trust',run_id=c['run_id'],policy_sha256=policy_sha,spec_sha256=spec_sha,prepare_sha256=prepare_sha,executor_source_sha256=executor_sha,commitment_payload_sha256=digest(canonical(c)),authorization_payload_sha256=digest(canonical(a)),trusted_timestamp_proven=False,dispatch_observed=False,outcome_blinding_proven=False,execution_authorized=False,P1_admission=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','verify']);
    for k in ('policy','policy-sha','spec','spec-sha','prepare','prepare-sha','executor','executor-sha','commitment','commitment-sha','authorization','authorization-sha','output'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();policy=parse(read(Path(a.policy),a.policy_sha,1024*1024));read(Path(a.spec),a.spec_sha);read(Path(a.prepare),a.prepare_sha);read(Path(a.executor),a.executor_sha)
    c=parse(read(Path(a.commitment),a.commitment_sha,1024*1024));auth=parse(read(Path(a.authorization),a.authorization_sha,1024*1024))
    result=verify(policy,a.policy_sha,a.spec_sha,a.prepare_sha,a.executor_sha,c,auth)
    result['input_files']={k:{'path':getattr(a,k),'sha256':getattr(a,k+'_sha')} for k in ('policy','spec','prepare','executor','commitment','authorization')}
    data=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();p=Path(a.output)
    need(p.is_absolute() and p.name.startswith(NAME+'_'),'own absolute output')
    if a.command=='check':
        with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:need(p.read_bytes()==data,'whole saved output mismatch')
    print(a.command,digest(data))

if __name__=='__main__':main()
