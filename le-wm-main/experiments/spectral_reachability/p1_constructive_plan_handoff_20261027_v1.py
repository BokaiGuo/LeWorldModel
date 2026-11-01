"""Explicit CPU-only planner.consume -> metadata-joined physical handoff.

Not a planner report and not an independent planner audit. Never read sourcepack,
certificate, witness, native state or physical outcomes. MAIN supplies evaluator
identity mapping separately. No model execution or simulator imports.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as planner

NAME='p1_constructive_plan_handoff_20261027_v1'
PHYSICAL='p1_constructive_plan_physical_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
STATUS='planner_verified_handoff_not_admitted'
FLOOR=200*1024**2
LIMIT=256*1024
require=planner.require


def read(path):return planner.read(path,LIMIT)
def sha(path):return planner.sha(Path(path))
def raw(value):return planner.canonical(value)
def digest(value):
    # Frozen physical.source.digest uses default JSON separators, not planner's compact canonical bytes.
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def hash_value(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'lowercase SHA256')
def external(path,expected):
    hash_value(expected);path=Path(path).absolute()
    require(sha(path)==expected,'external SHA mismatch: '+str(path));return path


def commands(value):
    a=np.asarray(value)
    require(a.shape==(10,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and (abs(a)<=1).all() and
        np.array_equal(a,a.astype(np.float32)),'literal float32 raw10x2, no rescale/clip/round')
    return deepcopy(value)


def closure():
    paths=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        Path(planner.__file__),Path(planner.encode.__file__)]
    return {str(p.resolve()):sha(p) for p in paths}


def build(planner_output,report_sha):
    """Read-only; consume owns original saved selection/source/model closure checks."""
    output=Path(planner_output).absolute();report=external(output/'report.json',report_sha)
    before=closure()
    consumed=planner.consume(output,report_sha)
    require(consumed['status']=='model_commands_verified_for_separate_MAIN_executor' and
        consumed['new_physics']==0 and consumed['P1_admission'] is False,'whole frozen planner consume release')
    plans=consumed['plans'];arms=list(planner.ARMS)
    require([(p['order'],p['arm']) for p in plans]==[(i,a) for i in range(2) for a in arms] and
        all(type(p['order']) is int for p in plans),'fixed two planner orders x both own arms')
    approved=list(planner.encode.APPROVED_INPUTS.values())
    require(len(approved)==2 and len(set(approved))==2,'this bridge exact two approved blind sources')
    exports={};rows=[];order_identities={}
    for plan in plans:
        require(plan['schema']==planner.NAME+'.model_plan' and plan['with_target'] is False and
            plan['physical_execution'] is False and plan['selection_uses_physics'] is False and
            plan['P1_admission'] is False and plan['whole_run_report_sha256']==report_sha,'model-only selected plan provenance')
        path=external(plan['task_export'],plan['task_sha256'])
        export=planner.read(path,2*1024**2)
        planner.validate_export_envelope(export)
        slot=export['source_slot'];blind=export['blind_source_sha256']
        require(type(slot) is int and slot in (0,1) and blind==approved[slot] and
            export['blind_sha256']==export['task_contract']['blind_source_sha256']==blind and
            export['task_id']==plan['task_id'],'explicit approved source_slot/blind/task metadata join')
        require(sha(path)==plan['task_sha256'],'task metadata TOCTOU')
        ident=(str(path),plan['task_sha256'],plan['task_id'],slot,blind)
        require(plan['order'] not in order_identities or order_identities[plan['order']]==ident,'both arms use identical own task')
        order_identities[plan['order']]=ident;exports[str(path)]=plan['task_sha256']
        hash_value(plan['selection_sha256'])
        rows.append(dict(source_slot=slot,arm=plan['arm'],commands=commands(plan['commands']),blind_sha256=blind,
            planner_order=plan['order'],task_export=str(path),task_sha256=plan['task_sha256'],task_id=plan['task_id'],
            planner_selection_sha256=plan['selection_sha256']))
    require(len(exports)==2 and {x[3] for x in order_identities.values()}=={0,1},'no duplicate or missing root/task')
    # Deterministic public metadata order only, NEVER sort by any predicted/native outcome.
    rows.sort(key=lambda r:(r['source_slot'],arms.index(r['arm'])))
    require(sha(report)==report_sha and closure()==before,'final report and bridge closure unchanged')
    for path,h in exports.items():require(sha(path)==h,'final export seal unchanged')
    return dict(schema=NAME+'.handoff',status=STATUS,errors=[],rows=rows,arms=arms,
        planner_output=str(output),planner_report_sha256=report_sha,task_bindings=exports,bridge_bindings=before,
        authority='explicit planner.consume release plus SHA-bound export metadata join; NOT original planner report or independent audit',
        selection_fixed_before_physics=True,physical_outcome_selection=False,sourcepack_read=False,certificate_read=False,
        new_model_calls=0,new_physics=0,P1_admission=False,global_freshness='unknown_calibration_only')


def verify_handoff(path,expected):
    path=external(path,expected);value=read(path)
    require(value==build(value['planner_output'],value['planner_report_sha256']),'handoff full saved reconstruction')
    require(sha(path)==expected,'handoff TOCTOU');return value


def selection(handoff_path,handoff_sha,evaluator_map,evaluator_sha):
    h=verify_handoff(handoff_path,handoff_sha);ep=external(evaluator_map,evaluator_sha);e=read(ep)
    fields={'schema','sourcepack','sourcepack_sha256','source_prepared','source_prepare_sha256','roots','wall_seconds'}
    require(set(e)==fields and e['schema']==NAME+'.evaluator_map','explicit separate evaluator-map schema')
    require(type(e['wall_seconds']) is int and 0<e['wall_seconds']<=600,'physical wall bound')
    require(isinstance(e['roots'],list) and len(e['roots'])==2 and all(set(r)=={'root_slot','root_sha256'} and
        type(r['root_slot']) is int and r['root_slot']==i for i,r in enumerate(e['roots'])),'literal evaluator root0/1 order')
    for r in e['roots']:hash_value(r['root_sha256'])
    for k in ('sourcepack_sha256','source_prepare_sha256'):hash_value(e[k])
    for k in ('sourcepack','source_prepared'):require(Path(e[k]).is_absolute(),'external absolute evaluator source paths')
    # Do NOT open these evaluator source paths. Physical.prepare authenticates them.
    result={k:deepcopy(e[k]) for k in fields-{'schema'}}
    result.update(schema=PHYSICAL+'.selection',arms=deepcopy(h['arms']),plans=[],selection_fixed_before_physics=True,physical_outcome_selection=False)
    for index,row in enumerate(h['rows']):
        base=f'/rows/{index}'
        plan=dict(root_slot=row['source_slot'],arm=row['arm'],commands=deepcopy(row['commands']),
            compiler_report=str(Path(handoff_path).absolute()),compiler_report_SHA=handoff_sha,
            compiler_schema=h['schema'],compiler_status=h['status'],commands_pointer=base+'/commands',
            source_slot_pointer=base+'/source_slot',blind_sha_pointer=base+'/blind_sha256',arm_pointer=base+'/arm')
        plan['selection_digest']=digest(plan);result['plans'].append(plan)
    require(sha(ep)==evaluator_sha and sha(handoff_path)==handoff_sha,'selection input TOCTOU')
    return result


def write_new(path,value):
    data=raw(value);path=Path(path).absolute()
    require(len(data)<=LIMIT and not path.exists() and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(data),'NEW <=256KiB output +200MiB reserve')
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=sha(path),bytes=len(data))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('handoff','verify','selection'))
    for key in ('planner-output','planner-report-sha','handoff','handoff-sha','evaluator-map','evaluator-map-sha','output'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='handoff':result=write_new(a.output,build(a.planner_output,a.planner_report_sha))
    elif a.command=='verify':result=dict(status=verify_handoff(a.handoff,a.handoff_sha)['status'])
    else:result=write_new(a.output,selection(a.handoff,a.handoff_sha,a.evaluator_map,a.evaluator_map_sha))
    print(raw(result).decode(),end='')


if __name__=='__main__':main()
