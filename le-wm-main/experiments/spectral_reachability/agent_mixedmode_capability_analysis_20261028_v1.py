"""Saved-only signed task response after full physical producer.consume.

No model, environment execution, selected probe/winner, metric fit or gate.
Common original root->goal directions are held fixed across before/after.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_capability_physical_20261028_v1 as physical

NAME='agent_mixedmode_capability_analysis_20261028_v1'
ROOT=physical.ROOT
REPORT_SHA='57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8'
PHYSICAL_SOURCE_SHA='21835721f0ee360dde66f1b0b754332438660ebac1fed636794fadbf6865a88c'
DEFAULT_OUTPUT=ROOT/'results/agent_mixedmode_capability_physical_20261028_v1_MAIN01'
CAP,FLOOR=1024**2,200*1024**2
require,sha,canonical=physical.require,physical.sha,physical.canonical
probe=physical.probe
SPACES=('joint_xy','agent_xy','object_xy','object_angle_wrapped')


def wrap(value):
    a=np.asarray(value,dtype=np.float64)
    return np.arctan2(np.sin(a),np.cos(a))


def feature(state,space):
    a=probe.vector(state)
    if space=='joint_xy':return a[:4]
    if space=='agent_xy':return a[:2]
    if space=='object_xy':return a[2:4]
    require(space=='object_angle_wrapped','fixed predeclared feature space')
    return a[4:5]


def difference(a,b,space):
    d=feature(a,space)-feature(b,space)
    return wrap(d) if space=='object_angle_wrapped' else d


def task_direction(native,space):
    d=difference(native['goal'],native['root_state'],space)
    n=float(np.linalg.norm(d))
    return None if n==0 else d/n


def scalar_projection(delta,u):
    return None if delta is None or u is None else float(np.dot(delta,u))


def subtract(a,b):
    return None if a is None or b is None else float(a-b)


def analyze(p,consumed):
    """Pure seam; production assemble ALWAYS full-calls physical.consume first.

    Missing evidence is retained as null and never substituted or optimized.
    """
    require(p['schema']==probe.NAME+'.prepare' and p['contract']==probe.contract() and
        len(p['tasks'])==4 and len(p['jobs'])==40 and len(consumed['rows'])==40,
        'complete original40 tasks/probes denominator')
    require(consumed['status'] in ('saved_probe_responses_not_admitted','partial_evidence_preserved_not_admitted') and
        consumed['P1_admission'] is False and consumed['new_neural_calls']==consumed['new_physics_calls']==0,
        'full own-consumer receipt, not raw physical row shortcut')
    rows=consumed['rows'];groups=[]
    for j,r in zip(p['jobs'],rows,strict=True):
        require(j['job_digest']==r['job_digest'] and j['task']==r['task'] and j['role']==r['role'] and
            j['probe_id']==r['probe_id'] and j['context_digest']==r['context_digest'], 'fixed identity join, no index guessing')
        require(r['status'] in ('completed','failed','native_stopped','unstarted') and
            (r['status']=='completed' or r['end_state'] is None),'partial endpoint remains unknown')
    for i,task in enumerate(p['tasks']):
        plan=task['plan'];native=task['native'];key=probe.ORDER[i]
        require((plan['root_slot'],plan['condition'])==key,'all four original identities')
        goal=probe.vector(native['goal']);root=probe.vector(native['root_state'])
        for role in ('before','after'):
            jobs=[j for j in p['jobs'] if j['task']==i and j['role']==role]
            require([j['probe_id'] for j in jobs]==[k for k,_ in probe.PROBES],'same ordered fixed five probes')
            rr=[rows[j['index']] for j in jobs]
            endpoints={r['probe_id']:probe.vector(r['end_state']) if r['status']=='completed' else None for r in rr}
            starts=[probe.vector(r['start_state']) for r in rr if r['start_verified']]
            require(not starts or all(a.tobytes()==starts[0].tobytes() for a in starts),'same causal start for five independent probes')
            for r in rr:
                if r['start_verified']:
                    require(r['goal']==goal.tolist() and r['root_state']==root.tolist() and
                        r['condition']==plan['condition'],'original native root/goal/condition, no surrogate target')
            start=None if not starts else starts[0]
            amplitude=float(probe.raw(jobs[1]['probe_commands'],5)[0,0])
            require(amplitude>0 and all(np.array_equal(probe.raw(j['probe_commands'],5),
                np.tile(np.asarray(d,dtype='<f4'),(5,1))) for j,(_,d) in zip(jobs,probe.PROBES,strict=True)),
                'unchanged literal signed probe dictionary')
            item=dict(task=i,root_slot=key[0],condition=key[1],role=role,
                task_id=plan['task_id'],selection_digest=plan['selection_digest'],context_digest=task['context_digest'],
                root_state=root.tolist(),goal=goal.tolist(),start_state=None if start is None else start.tolist(),
                probe_amplitude_raw=amplitude,elapsed_public_steps_before_probe=10+(5 if role=='after' else 0),
                missing=[r['probe_id'] for r in rr if r['status']!='completed'],
                evidence=[dict(probe_id=r['probe_id'],job_digest=r['job_digest'],status=r['status'],
                    attempted=r['attempted'],completed=r['completed'],error=deepcopy(r['error']),
                    last_return=deepcopy(r['last_return'])) for r in rr],spaces={})
            # Raw body velocities/angles/forces are kept separately; native7
            # velocity entries alone do not summarize the block contact state.
            reference=next((r for r in rr if r['frames']),None)
            item['start_snapshot']=None if reference is None else deepcopy(reference['frames'][0]['pre']['snapshot'])
            for space in SPACES:
                u=task_direction(native,space);z=endpoints['zero'];responses=[]
                for probe_id,_ in probe.PROBES:
                    end=endpoints[probe_id]
                    delta=None if end is None or z is None else difference(end,z,space)
                    responses.append(dict(probe_id=probe_id,zero_relative_delta=None if delta is None else delta.tolist(),
                        signed_original_task_projection=scalar_projection(delta,u)))
                cols=[];asym=[];cut=[]
                for axis in ('x','y'):
                    plus,minus=endpoints['+'+axis],endpoints['-'+axis]
                    d=None if plus is None or minus is None else difference(plus,minus,space)/(2*amplitude)
                    cols.append(None if d is None else d.tolist())
                    asym.append(None if plus is None or minus is None or z is None else
                        (difference(plus,z,space)+difference(minus,z,space)).tolist())
                    cut.append(None if plus is None or minus is None or space!='object_angle_wrapped' else
                        bool(abs(plus[4]-minus[4])>=np.pi))
                drift=None if z is None or start is None else difference(z,start,space)
                item['spaces'][space]=dict(original_task_direction=None if u is None else u.tolist(),
                    direction_status='zero_original_residual_no_direction' if u is None else 'fixed_original_root_to_goal',
                    finite_responses=responses,zero_drift=None if drift is None else drift.tolist(),
                    central_secant_columns=cols,central_secant_units='rad/raw_held_command' if space=='object_angle_wrapped' else 'px/raw_held_command',
                    signed_original_task_secants=[scalar_projection(c,u) for c in cols],
                    asymmetry_columns=asym,angle_branch_cut=cut,
                    unsigned_spectrum=None,full_task_success_inferred=False)
            groups.append(item)
    comparisons=[]
    for i in range(4):
        before,after=groups[2*i:2*i+2];spaces={}
        for space in SPACES:
            b,a=before['spaces'][space],after['spaces'][space]
            spaces[space]=dict(signed_response_changes=[dict(probe_id=x['probe_id'],
                after_minus_before=subtract(x['signed_original_task_projection'],y['signed_original_task_projection']))
                for x,y in zip(a['finite_responses'],b['finite_responses'],strict=True)],
                signed_secant_changes=[subtract(x,y) for x,y in zip(a['signed_original_task_secants'],b['signed_original_task_secants'],strict=True)])
        comparisons.append(dict(task=i,root_slot=before['root_slot'],condition=before['condition'],spaces=spaces,
            time_confound=dict(controlled=False,elapsed_before=10,elapsed_after=15,
                start_state_before=before['start_state'],start_state_after=after['start_state'],
                start_snapshot_before=before['start_snapshot'],start_snapshot_after=after['start_snapshot'],
                zero_drift_before={s:before['spaces'][s]['zero_drift'] for s in SPACES},
                zero_drift_after={s:after['spaces'][s]['zero_drift'] for s in SPACES}),
            causal_capability_gain='unknown',P1_admission=False))
    result=dict(schema=NAME+'.report',status='saved_signed_response_described_not_admitted',
        denominators=dict(known_roots=2,root_conditions=4,roles=8,probes=40,fresh_roots=0),
        rows=groups,comparisons=comparisons,counts=deepcopy(consumed['counts']),errors=deepcopy(consumed.get('errors',[])),
        new_neural_calls=0,new_physics_calls=0,training=False,Qfit=False,P1_admission=False,
        independent_confirmation=False,equal_total_compute=False,
        limitations=['same original task direction is shared, not after residual refit',
            'signed projection is directional response, not Euclidean terminal improvement or native success',
            'jointXY includes agent and object; object-only projection cannot establish full task failure',
            'angle radians reported separately, no fitted mixing with pixels or velocity',
            'zero-relative does not control elapsed time, state, velocity, contact or unexported solver cache',
            'five constant XY controls and one amplitude, not all10 coordinates or global controllability',
            'do not infer trajectory contact from terminal snapshots',
            'no probe/root selection, thresholds, p-values or theory rejection'])
    canonical(result)
    return result


def assemble(output,report_sha):
    require(sha(physical.__file__)==PHYSICAL_SOURCE_SHA,'frozen physical source before consume')
    require(probe.hash_value(report_sha)==REPORT_SHA,'fixed actual40 report only')
    directory=Path(output).resolve()
    consumed=physical.consume(directory,report_sha)  # Mandatory full actual consumer.
    require(sha(directory/'report.json')==report_sha,'actual report unchanged after full consume')
    report=probe.read(directory/'report.json')
    binding=report['execution_binding']
    require(sha(binding['probe_prepared'])==binding['probe_prepare_sha256'],'bound actual probe prepare')
    p=probe.read(binding['probe_prepared'])
    result=analyze(p,consumed)
    result.update(physical_report=str(directory/'report.json'),physical_report_sha256=report_sha,
        physical_prepare_sha256=report['executor_prepare_sha256'],probe_prepare_sha256=binding['probe_prepare_sha256'],
        physical_source_sha256=PHYSICAL_SOURCE_SHA,journal_sha256=report['journal']['sha256'],
        full_producer_consume_status=consumed['status'],
        own_file_sha256={str(f.resolve()):sha(f) for f in [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]})
    require(sha(directory/'report.json')==report_sha and sha(physical.__file__)==PHYSICAL_SOURCE_SHA,'final input seals')
    require(len(canonical(result))<=CAP,'bounded1MiB arithmetic report')
    return result


def write_new(path,result):
    path=Path(path).absolute();raw=canonical(result)
    require(path.name.startswith(NAME+'_') and path.suffix=='.json' and len(raw)<=CAP and
        shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive own1MiB output and200MiB floor')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'complete report write');f.flush();os.fsync(f.fileno())
    physical.adapter.paired.fsync_dir(path.parent)
    return dict(output=str(path),sha256=sha(path),bytes=len(raw))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-output',default=str(DEFAULT_OUTPUT))
    parser.add_argument('--physical-report-sha',default=REPORT_SHA)
    parser.add_argument('--destination')
    a=parser.parse_args(argv);result=assemble(a.physical_output,a.physical_report_sha)
    print(canonical(result if a.destination is None else write_new(a.destination,result)).decode())


if __name__=='__main__':main()
