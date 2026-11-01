"""Executable protocol preparation and missing-asset intake, never model dispatch.

Reuse sealed readiness instead of duplicating its workflow checker. A manifest
is an execution handoff, NOT a result table or permission to load a checkpoint.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from experiments.spectral_reachability import agent_baseline_launch_readiness_20261028_v1 as readiness

NAME='agent_baseline_protocol_bridge_20261028_v1'
ROOT=readiness.ROOT
READINESS_SHA='6056d61e1463a13db6a20d28467196558d1ad92de4790c25c2f8f08d979bb042'
CAP,FLOOR=2*1024**2,200*1024**2
METHODS=('GRASP','Hi-LeWM','LeWM','PlanWAM')
CELLS=('grasp:S4.T1.3:5:1','hilewm:A3.T4.3:15:7','lewm_v3:A7.T10.6:1:1','planwam:S4.T1.6:16:11')
SUPPORT={
 'moving_grasp':'virtual-state/soft-dynamics joint optimization plus serial action sync; not GD relabelled',
 'matched_sampling':'CEM and local nominal-MPPI; 3x24 H2 candidates; only these two share 144 API/sample per root',
 'moving_gradient':'open-loop gradient optimizer; NOT closed-loop MPC or all paper gradient solvers'}
# Exact roles for a future material handoff. Role presence alone NEVER enables inference.
ROLES={
 'GRASP':('official_solver_revision','checkpoint_config_manifest','dataset_split_manifest','frozen_normalizer','paper_protocol_budget','license'),
 'Hi-LeWM':('verified_archive_receipt','high_object','source_dependency_lock','embedded_low_identity','dataset_split_manifest','frozen_normalizer','hierarchical_protocol','license'),
 'LeWM':('official_source_revision','checkpoint_config_manifest','dataset_split_manifest','frozen_normalizer','table10_solver_budget','license'),
 'PlanWAM':('official_source_revision','checkpoint_config_manifest','camera_ego_split_manifest','evaluator_revision','inference_protocol','license')}


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()+b'\n'
def digest(raw):return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON '+x)))


def read(path):
    with Path(path).open('rb') as f:raw=f.read(CAP+1)
    need(len(raw)<=CAP,'bounded metadata <=2MiB');return decode(raw)


def task_protocol(e,slot,task_sha):
    """Pure identity extraction from an externally pinned original task export."""
    t,c=e['task'],e['task_contract'];root=t['root_context']
    need(type(e['source_slot']) is int and e['source_slot']==c['source_slot']==slot,'literal source slot')
    need(e['with_target'] is c['with_target'] is False and e['P1_admission'] is False,'original off/calibration task')
    need(t['environment']=='swm/PushT-v1' and t['horizon']==2 and t['packet_steps']==5 and t['control_dim']==2 and
         t['latent_dim']==192 and t['feedback']=='open_loop_model_only','fixed H2 raw10 task')
    need(root['layout']=='states_with_between_state_actions_v1' and root['context_length']==3 and
         root['latent_dim']==192 and root['action_dim']==10,'real RGB3 / two incoming packets')
    def shape(v,dims):
        if not dims:return type(v) in (int,float) and math.isfinite(v)
        return isinstance(v,list) and len(v)==dims[0] and all(shape(x,dims[1:]) for x in v)
    need(shape(root['latent_history'],(3,192)) and shape(root['action_history'],(2,10)) and
         shape(t['goal'],(192,)) and shape(t['projection'],(192,192)),'full finite context/goal/Q')
    need(all(t['projection'][i][j]==int(i==j) for i in range(192) for j in range(192)) and
         c['metric']=='identity_native192_no_fit','identity Q; not fitted readout')
    need(t['root_raw_action_history']==[[0.10000000149011612,0.]]*5+[[0.,0.10000000149011612]]*5 and
         c['observed_steps']==[0,5,10],'actual nonzero moving history; no padding')
    need(t['action_mean']==c['normalization']['mean'] and t['action_std']==c['normalization']['std'] and
         len(t['action_mean'])==2 and len(t['action_std'])==2 and all(x>0 and math.isfinite(x) for x in t['action_std']),
         'same original normalizer, not fit')
    need(t['packet_logit_cap']==.75*math.sqrt(5),'original packet logit domain')
    return dict(source_slot=slot,task_sha256=task_sha,task_id=e['task_id'],blind_sha256=e['blind_sha256'],
        task_contract=deepcopy(c),model_tensor_sha256=readiness.MODEL_SHA,
        root_context_sha256=digest(canonical(root)),goal_sha256=digest(canonical(t['goal'])),
        raw_history_sha256=digest(canonical(t['root_raw_action_history'])),Q_sha256=digest(canonical(t['projection'])),
        action_domain=dict(packet_steps=5,packets=2,control_dim=2,packet_logit_cap=t['packet_logit_cap'],map='float32 tanh then raw10x2'),
        environment=t['environment'],feedback=t['feedback'],native_success='norm(state7-goal7 first4)<20 and min(abs(delta_angle),2pi-abs(delta_angle))<pi/9',
        native_distance='norm(all seven including velocity); no object-only replacement',
        Q='float64 mean192 squared error; NOT native task success',freshness='known_root_calibration_only')


def request(method):
    need(method in METHODS,'fixed supported method')
    return dict(schema=NAME+'.asset_request',method=method,variant='Hi-LeWM base, NOT -C' if method=='Hi-LeWM' else 'paper_native',
        required_roles=list(ROLES[method]),records=[dict(role=r,path=None,sha256=None) for r in ROLES[method]],
        transport_only=True,unknown_fields_forbid=True,inference_authorized=False,
        additional_semantic_gates=['MAIN source/license/trusted-loading review','checkpoint/module/config identity',
            'frozen dataset/normalizer provenance; no refit','exact selected protocol and budget','MAIN CUDA-only runtime implementation/validation'],
        notes=('Hi-LeWM-C additionally needs empirical macro bank/indices/encoder/data identity; not requested as base.' if method=='Hi-LeWM' else
               'PlanWAM driving camera/evaluator cannot be replaced by PushT.' if method=='PlanWAM' else
               'Existing local adaptation does not certify paper-native protocol.'))


def validate_records(method,records,hash_file):
    need(method in METHODS and isinstance(records,list),'supported method and record list')
    need(all(isinstance(x,dict) and set(x)=={'role','path','sha256'} for x in records),'exact asset record fields')
    roles=[x['role'] for x in records];need(len(roles)==len(set(roles)) and set(roles)==set(ROLES[method]),'exact roles, no missing/extra/duplicate')
    errors=[];verified=[]
    for record in records:
        try:
            p,h=record['path'],record['sha256']
            need(isinstance(p,str) and Path(p).is_absolute() and '..' not in Path(p).parts,'absolute nontraversing path')
            need(not any(s in p.lower() for s in ('.partial','.part','.download','.tmp')),'final file, not incomplete transfer')
            need(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h),'literal external SHA256 required')
            need(hash_file(p)==h,'asset bytes SHA mismatch');verified.append(deepcopy(record))
        except (OSError,ValueError) as exc:errors.append(dict(role=record['role'],error=str(exc)))
    return dict(schema=NAME+'.asset_intake',method=method,status='materials_verified_not_runtime_ready' if not errors else 'blocked',
        verified=verified,errors=errors,semantic_review_complete=False,inference_authorized=False,official_protocol_ready=False,
        local_result=None,P1_admission=False,scientific_admission=False)


def hash_material(path):
    # Same fd throughout, no pickle/load/import/extraction. Explicit candidate only.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno());need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=4*1024**3,'regular nonempty <=4GiB material')
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
        after=os.fstat(f.fileno());current=os.stat(path,follow_symlinks=False)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(identity(before)==identity(after)==identity(current),'material changed during intake')
    return h.hexdigest()


def assemble(preflight,prepared,reports,protocols,author_rows):
    """No numeric outcomes enter eligibility; method order fixed before reads."""
    need(not preflight['shared_errors'] and len(protocols)==2,'shared protocol authority')
    rows=[]
    for w,p,r in zip(readiness.WORKFLOWS,prepared,reports,strict=True):
        fw=next(x for x in preflight['local_workflows'] if x['workflow']==w.key)
        need(fw['status']=='ready_for_MAIN_dry_preflight','blocked local workflow '+w.key)
        need(p['expected_model_hash']==readiness.MODEL_SHA and p['contract']==r['contract'],'same frozen model/contract')
        slots=p['slots'];need(len(slots)==2,'both roots, no selection')
        for slot,identity in zip(slots,protocols,strict=True):
            need(type(slot['source_slot']) is int and all(slot[k]==identity[k] for k in ('source_slot','task_sha256','task_id','blind_sha256')),'same actual task identity')
        c=p['contract']
        need(c['with_target'] is c['training'] is c['P1_admission'] is False,'fixed local scope')
        if w.key=='moving_gradient':
            need(c['logits_shape']==[2,10] and c['raw_commands_shape']==[10,2] and c['feedback_MPC'] is False,'gradient explicit H2 contract')
            arms=[c['arm']]
        else:
            need(c['horizon']==2,'H2 contract');arms=c['arms']
        rows.append(dict(workflow=w.key,method=w.method,arms=deepcopy(arms),
            source_sha256=w.source_sha,prepared=fw['prepared'],prepare_sha256=w.prepare_sha,
            report=fw['saved_report'],report_sha256=w.report_sha,contract=deepcopy(p['contract']),
            actual_resource_receipt=deepcopy(r['resources']),predict_API_receipt=deepcopy(r.get('predict_API_resources')),
            search_support=SUPPORT[w.key],task_identity_sha256=[digest(canonical(x)) for x in protocols],
            main_saved_commands=deepcopy(fw['main_dry_preflight']),own_strict_consumer_executed=False,
            local_result_values=None,local_scope='known2 moving calibration, not paper replication',official_equivalence=False))
    need([x['id'] for x in author_rows]==list(CELLS),'fixed author cells not posthoc selection')
    return dict(schema=NAME+'.protocol_manifest',local_workflows=rows,shared_tasks=deepcopy(protocols),
        same_saved_task_identity=True,selected_action_domain_verification='requires each ownconsume; not claimed by metadata bridge',
        same_search_support=False,equal_predict_budget_GRASP_vs_LeWM=False,equal_FLOPs=False,
        forward_matched_pair=['CEM','kernel_faithful_mppi_nominal'],author_reported_context_only=deepcopy(author_rows),
        author_values_are_local_results=False,cross_paper_ranking_allowed=False,
        missing_asset_requests=[request(m) for m in METHODS],
        local_adaptations_not_official=['GRASP','LeWM'],unavailable_local_methods=['Hi-LeWM','PlanWAM'],
        all_methods_denominator=4,neural_calls=0,physics_calls=0,network_calls=0,launch_authorized=False,
        strict_next_step='MAIN runs existing verify + ownconsume commands; no rerun needed for existing results',
        P1_admission=False,scientific_admission=False)


def build():
    files=readiness.Files(ROOT);errors=[]
    need(readiness.check(files,Path(readiness.__file__),READINESS_SHA,errors),'sealed readiness helper')
    preflight=readiness.build(files);ps=[];rs=[]
    for w in readiness.WORKFLOWS:
        ps.append(files.json(files.path(w.prepared)));rs.append(files.json(files.path(w.output)/'report.json'))
    protocols=[]
    for slot in ps[0]['slots']:
        path=Path(slot['task_export']);need(files.digest(path)==slot['task_sha256'],'task export SHA')
        with gzip.open(path,'rb') as f:raw=f.read(CAP+1)
        need(len(raw)<=CAP,'decoded task bound');e=decode(raw)
        protocols.append(task_protocol(e,slot['source_slot'],slot['task_sha256']))
    need([x['task_sha256'] for x in protocols]==readiness.TASK_SHAS,'fixed two original tasks')
    need(readiness.check(files,files.path(readiness.REPORTED),readiness.REPORTED_SHA,errors),'pinned author registry')
    registry=files.json(files.path(readiness.REPORTED));authors=[]
    for key in CELLS:
        matches=[x for x in registry['rows'] if x['id']==key];need(len(matches)==1,'unique author locator');authors.append(matches[0])
    out=assemble(preflight,ps,rs,protocols,authors)
    bindings={str(p):h for p,h in files.hashes.items()}
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f.resolve())]=files.digest(f)
    for path,h in bindings.items():need(hash_material(path)==h,'final bound material SHA')
    out.update(bindings_sha256=bindings,bytes_hashed_initial=files.bytes_hashed)
    return out


def write_new(path,value):
    path=Path(path).absolute();raw=canonical(value)
    need(len(raw)<=CAP and not path.exists() and not path.is_symlink() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive <=2MiB / 200MiB reserve')
    with path.open('xb') as f:need(f.write(raw)==len(raw),'full manifest write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=hash_material(path),bytes=len(raw),launch_authorized=False)


def verify(path,expected):
    need(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) and hash_material(path)==expected,'external protocol manifest SHA')
    p=read(path);need(p==build() and hash_material(path)==expected,'entire reconstructed protocol/asset closure')
    return dict(schema=NAME+'.verified',local_workflows=len(p['local_workflows']),shared_tasks=2,all_methods_denominator=4,
        missing_local_methods=p['unavailable_local_methods'],launch_authorized=False,P1_admission=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('check','prepare','verify','request','intake'))
    parser.add_argument('--output');parser.add_argument('--manifest');parser.add_argument('--sha256');parser.add_argument('--method',choices=METHODS)
    a=parser.parse_args(argv)
    if a.command in ('check','prepare'):
        result=build()
        if a.command=='prepare':need(a.output is not None,'exclusive output required');result=write_new(a.output,result)
    elif a.command=='verify':result=verify(a.manifest,a.sha256)
    elif a.command=='request':
        result=request(a.method)
        if a.output:result=write_new(a.output,result)
    else:
        need(hash_material(a.manifest)==a.sha256,'external intake request SHA');p=read(a.manifest)
        need(set(p)==set(request(a.method)) and all(p[k]==request(a.method)[k] for k in p if k!='records'),'fixed intake schema; only records filled')
        result=validate_records(a.method,p['records'],hash_material)
        if a.output:write_new(a.output,result)
    print(canonical(result).decode(),end='')
    return 1 if result.get('status')=='blocked' else 0


if __name__=='__main__':raise SystemExit(main())
