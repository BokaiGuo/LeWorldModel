"""Pre-dispatch proposal and conditional external-signature check; never execute."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from experiments.spectral_reachability import agent_p1_external_temporal_binding_20261029_v1 as external

NAME='agent_p3_capability_prospective_binding_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md'
PROTOCOL_SHA='b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf'
SIGNATURE_SOURCE_SHA='648917c2c05f76ce517855ba564ac25b184a630dff451db2a595244938c85b78'
OLD_ENDPOINT_TASKS_SHA='1ff29caa35acc862ce63d64f157024cb61d3a7e1ad76ad1d43f7ba11a09d8e7d'


def need(ok,message):
    if not ok: raise ValueError(message)


def byte_hash(path, expected, cap=128*1024**2):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'canonical bounded input')
    need(isinstance(expected,str) and re.fullmatch(r'[0-9a-f]{64}',expected),'explicit SHA')
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    need(h.hexdigest()==expected,'input byte SHA mismatch')
    return dict(path=str(p),sha256=expected)


def contract():
    return dict(endpoint_order=[[0,'C'],[0,'Z'],[0,'P'],[1,'Z'],[1,'P']],known_roots=2,
        cp5_signals=2,cp10_signals=5,J=7,J_shape=[192,10],VJP=1344,AD_primal=7,
        predict_API=24,context_tokens=72,solve=7,program_compile=5,replay=10,Q_calls=22,
        stage_C_branches=10,logical_C_slots=4,logical_C_controls=8,
        stage_A_physics=dict(constructor=4,close=4,reset=8,set_state=16,goal_render=8,goal_bind=4,goal_setter=12,public_step=80,physics_step=816,explicit_render=264,png=264),
        stage_C_physics=dict(constructor=10,close=10,reset=20,set_state=40,goal_render=20,goal_bind=10,goal_setter=30,public_step=250,physics_step=2540,explicit_render=810,png=810),
        gpu_allocated_cap=3*1024**3,gpu_reserved_cap=4*1024**3,output_cap=2*1024**3,final_reserve=256*1024**2,wall_seconds=1800,
        native_comparison='D15(Z2)-D15(U); original goal7 and native success unchanged',
        selection='existing fixed protocol only; no outcome selection or substituted witness',
        FLOPs=None,P1_admission=False,P3_admission=False,full_C3_demonstrated=False,
        fresh_roots_certified=False,scientific_admission=False)


def proposal(run_id,cohort,cohort_sha,prepare,prepare_sha,executor,executor_sha):
    need(isinstance(run_id,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',run_id),'run_id')
    byte_hash(PROTOCOL,PROTOCOL_SHA)
    signer=byte_hash(Path(external.__file__).resolve(),SIGNATURE_SOURCE_SHA)
    own=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    source_closure={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in own}
    c=byte_hash(cohort,cohort_sha);p=byte_hash(prepare,prepare_sha);e=byte_hash(executor,executor_sha)
    need(cohort_sha!=OLD_ENDPOINT_TASKS_SHA,'known observed endpoint task bundle cannot be reclassified as future')
    need(len({cohort_sha,prepare_sha,executor_sha})==3,'separate bound artifacts')
    return dict(schema=NAME+'.proposal',run_id=run_id,protocol=dict(path=str(PROTOCOL),sha256=PROTOCOL_SHA),
        future_cohort=c,future_prepare=p,future_executor=e,signature_verifier=signer,source_closure=source_closure,contract=contract(),
        status='blocked_external_witness_future_runtime_and_freshness_unverified',execution_authorized=False,
        trusted_timestamp_proven=False,dispatch_observed=False,outcome_blinding_proven=False,
        future_cohort_authority_verified=False,future_runtime_abi_verified=False)


def signed_check(proposed,proposal_sha,policy,policy_sha,commitment,authorization):
    need(proposed['schema']==NAME+'.proposal' and proposed['contract']==contract() and proposed['execution_authorized'] is False,'proposal schema and contract')
    byte_hash(PROTOCOL,PROTOCOL_SHA)
    for k in ('future_cohort','future_prepare','future_executor'):
        x=proposed[k];need(set(x)=={'path','sha256'},'descriptor');byte_hash(x['path'],x['sha256'])
    rebuilt=proposal(proposed['run_id'],proposed['future_cohort']['path'],proposed['future_cohort']['sha256'],
        proposed['future_prepare']['path'],proposed['future_prepare']['sha256'],proposed['future_executor']['path'],proposed['future_executor']['sha256'])
    need(proposed==rebuilt,'complete proposal reconstruction')
    check=external.verify(policy,policy_sha,proposal_sha,proposed['future_prepare']['sha256'],proposed['future_executor']['sha256'],commitment,authorization)
    need(check['run_id']==proposed['run_id'],'signed run identity')
    return dict(schema=NAME+'.signed_preflight',proposal_sha256=proposal_sha,signature_check=check,
        status='conditional_signatures_only_blocked',execution_authorized=False,trusted_timestamp_proven=False,
        dispatch_observed=False,outcome_blinding_proven=False,future_cohort_authority_verified=False,
        future_runtime_abi_verified=False,independent_confirmation=False,P1_admission=False,P3_admission=False,full_C3_demonstrated=False)


def encoded(obj):return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def saved(path,sha):return external.parse(external.read(Path(path),sha))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('proposal','check','verify'))
    for k in ('run-id','cohort','cohort-sha','prepare','prepare-sha','executor','executor-sha','proposal','proposal-sha','policy','policy-sha','commitment','commitment-sha','authorization','authorization-sha','output','output-sha'):
        ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='proposal':v=proposal(a.run_id,a.cohort,a.cohort_sha,a.prepare,a.prepare_sha,a.executor,a.executor_sha)
    else:
        proposed=saved(a.proposal,a.proposal_sha);policy=saved(a.policy,a.policy_sha)
        c=saved(a.commitment,a.commitment_sha);auth=saved(a.authorization,a.authorization_sha)
        v=signed_check(proposed,a.proposal_sha,policy,a.policy_sha,c,auth)
        v['inputs']={k:dict(path=getattr(a,k),sha256=getattr(a,k+'_sha')) for k in ('proposal','policy','commitment','authorization')}
    data=encoded(v);p=Path(a.output)
    need(p.is_absolute() and p.name.startswith(NAME+'_') and not p.is_symlink(),'own absolute output')
    if a.command=='verify':need(external.read(p,a.output_sha)==data,'whole output bytes')
    else:
        need(a.output_sha is None,'new output')
        with p.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(p.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(fd)
        finally:os.close(fd)
    print(a.command,hashlib.sha256(data).hexdigest())


if __name__=='__main__':main()
