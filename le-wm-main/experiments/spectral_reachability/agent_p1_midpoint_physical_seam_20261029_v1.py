"""Saved midpoint-to-native execution seam; never executes physical actions."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_p1_midpoint_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as physical

NAME='agent_p1_midpoint_physical_seam_20261029_v1'
ROOT=host.ROOT
require,sha,raw=host.require,host.sha,host.raw
digest=physical.digest
ORDER=physical.ORDER
HOST_SHA='7f5c804dad8233bb9697a4c2cf16b1c71f26e39afeac277e80ac591d4900746e'


def contract():
    return dict(schema=NAME+'.contract',fixed_denominator=4,known_roots=2,fresh_roots=0,
        order=[list(x) for x in ORDER],raw_steps=10,guard_step=5,
        selection='exact verified selected_iteration; null is abstention; never substitute original winner',
        guard='rerun original prefix5 from same root; require complete original prefix51 triplets and cp5 state/snapshot equality',
        source_status='model strict complete only; physical_release remains false',
        actual_new_compute=0,actual_new_physics=0,physical_release=False,P1_admission=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def assemble(arguments,p,r,u,bp,br):
    """Pure join: caller MUST have reconstructed host.consume and native ancestry."""
    require(p['schema']==host.NAME+'.prepare' and p['contract']==host.contract() and p['production_ready'] is True,
            'real host preparation schema')
    require(r['schema']==host.NAME+'.report' and r['status']=='complete_model_only' and r['errors']==[]
            and r['fixed_denominator']==4 and r['prepared']==arguments['host_prepared']
            and r['prepare_sha256']==arguments['host_prepare_sha'], 'complete exact host report')
    require(u['schema']==host.NAME+'.consume' and u['status']=='verified_model_only' and u['complete'] is True
            and u['report_sha256']==arguments['host_report_sha'] and u['fixed_denominator']==4 and u['known_roots']==2
            and raw(u['rows'])==raw(r['rows']) and len(u['audits'])==4, 'complete actual saved consumer')
    require(all(x[k] is False for x in (r,u) for k in ('P1_admission','freshness','physical_release')),
            'do not manufacture physical release or admission')
    require(u['actual_new_compute']=={k:host.F.caps(104,32) for k in ('attempted','completed')}, 'actual complete native accounting')
    c=p['original_prepare'];physical.base.validate_execution(bp)
    require(br['schema']==physical.base.NAME+'.report' and br['status']==physical.base.RECORDED and not br['errors']
            and len(br['rows'])==len(bp['plans'])==len(c['rows'])==len(r['rows'])==4
            and c['original_physical_authority']==bp['planner_authority'], 'exact inherited physical source')
    plans=[]
    for i,(spec,row,a,old,actual) in enumerate(zip(c['rows'],r['rows'],u['audits'],bp['plans'],br['rows'],strict=True)):
        root,condition=ORDER[i];q=spec['request'];task=host.core.validate_task(q['task'])
        require(type(spec['index']) is type(row['index']) is int and spec['index']==row['index']==i
                and (spec['root_slot'],spec['condition'])==(root,condition) and row['status']=='sealed'
                and actual['index']==i and actual['plan']==old and actual['status']=='horizon_completed'
                and actual['initial_exact'] is True,'all slot/source identities')
        native=bp['source_roots'][root]
        require(actual['root_state']==native['root_state'] and actual['root_sha256']==native['root_sha256']
                and actual['goal']==native['goal'] and actual['goal_pose']==native['goal_pose']
                and actual['goal_pixel_sha256']==native['goal_pixels'][condition]
                and old['model_task_binding']==task['binding'],'same original root, goal, condition and task')
        cps=[x for x in actual['checkpoints'] if x['step']==5]
        require(len(cps)==1 and actual['prefix51']['complete'] is True and len(actual['prefix51']['frames'])==51
                and cps[0]['state']==cps[0]['snapshot']['state'],'complete original prefix/cp5 guard')
        result=row['result'];chosen=result['selected_iteration']
        require(a['selected_iteration']==chosen and raw(a['selected_commands'])==raw(result['selected_commands']), 'consumer and saved selected raw identity')
        baseline=physical.mixed.raw10(old['commands']);prefix=physical.mixed.raw10(q['incumbent']['prefix']+q['incumbent']['commands'])[:5]
        require(prefix.tobytes()==baseline[:5].tobytes() and host.F.bits(prefix)==q['incumbent']['prefix_bits'],'literal original prefix bytes')
        reference=dict(index=i,complete=True,raw_prefix_bits=host.F.bits(prefix),frames=deepcopy(actual['prefix51']['frames']),checkpoint=deepcopy(cps[0]))
        if chosen is None:
            require(result['status']=='abstain' and result['selected_commands'] is None,'explicit null abstention')
            command=None;bits=None;raw_sha=None;status='abstain'
        else:
            require(type(chosen) is int and 1<=chosen<=8 and result['status']=='model_candidate','exact eligible winner only')
            command=physical.mixed.raw10(result['selected_commands'])
            require(command[:5].tobytes()==prefix.tobytes()
                    and host.F.bits(command[5:])==result['original']['numeric']['iterates'][chosen]['raw_bits'], 'full10 exact selected action bits')
            bits=host.F.bits(command);raw_sha=hashlib.sha256(command.tobytes()).hexdigest();command=command.tolist();status='model_candidate_only'
        plan=dict(index=i,root_slot=root,condition=condition,status=status,selected_iteration=chosen,commands=command,
            raw_float32_bits=bits,raw_float32_sha256=raw_sha,task=deepcopy(task),task_binding=deepcopy(task['binding']),
            context_sha256=host.digest(q['observed_context']),native_root_sha256=native['root_sha256'],
            native_goal=deepcopy(native['goal']),native_goal_pose=deepcopy(native['goal_pose']),
            goal_pixel_sha256=native['goal_pixels'][condition],prefix_reference=reference,
            physical_release=False,P1_admission=False)
        plan['plan_sha256']=digest(plan);plans.append(plan)
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,
        fixed_denominator=4,known_roots=2,host_receipt_complete=True,physical_release=False,P1_admission=False,
        note='model candidate preparation only; physical execution not authorized by this record')


def build(host_output,host_report_sha,host_prepared,host_prepare_sha,host_receipt,host_receipt_sha):
    args=dict(host_output=host_output,host_report_sha=host_report_sha,host_prepared=host_prepared,
              host_prepare_sha=host_prepare_sha,host_receipt=host_receipt,host_receipt_sha=host_receipt_sha)
    require(sha(host.__file__)==HOST_SHA,'exact reviewed host source')
    for k,v in args.items():
        if k.endswith('_sha'):host.core.hash_value(v)
        else:physical.mixed.absolute(v)
    out=Path(host_output)
    for f,h in ((out/'report.json',host_report_sha),(host_prepared,host_prepare_sha),(host_receipt,host_receipt_sha)):
        require(sha(f)==h,'actual external SHA')
    u=host.consume(out,host_report_sha)
    require(raw(u)==raw(host.signal.read(host_receipt,64*1024**2)),'complete persisted consumer equality')
    r=host.signal.read(out/'report.json',64*1024**2);p=host.signal.read(host_prepared,64*1024**2)
    bp,br=physical.reference(p['original_prepare'])
    result=assemble(args,p,r,u,bp,br)
    bindings=deepcopy(p['bindings'])
    for f,h in {host_prepared:host_prepare_sha,str(out/'report.json'):host_report_sha,host_receipt:host_receipt_sha,
                **{str(out/n):h for n,h in r['artifact_sha256'].items()}}.items():
        require(f not in bindings or bindings[f]==h,'no source conflict');bindings[f]=h
    for f in [*own_files(),Path(physical.__file__).resolve(),Path(physical.base.__file__).resolve()]:bindings[str(f)]=sha(f)
    host.F.check_bindings(bindings);result['bindings']=bindings
    return result


def verify(path,expected):
    host.core.hash_value(expected);require(sha(path)==expected,'prepare hash')
    p=host.signal.read(path,128*1024**2)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'full saved authority rebuild')
    return p


def write(path,value):
    path=Path(physical.mixed.absolute(path));payload=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink()
            and len(payload)<=128*1024**2 and shutil.disk_usage(path.parent).free>=200*1024**2+len(payload),'exclusive bounded output')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full write');f.flush();os.fsync(f.fileno())
    physical.paired.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify'))
    for k in ('host-output','host-report-sha','host-prepared','host-prepare-sha','host-receipt','host-receipt-sha','output','prepared','prepare-sha'):q.add_argument('--'+k)
    a=q.parse_args()
    result=write(a.output,build(a.host_output,a.host_report_sha,a.host_prepared,a.host_prepare_sha,a.host_receipt,a.host_receipt_sha)) if a.command=='prepare' else dict(verified=True,physical_release=verify(a.prepared,a.prepare_sha)['physical_release'])
    print(raw(result).decode(),flush=True)


if __name__=='__main__':main()
