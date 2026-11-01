"""Saved signed comparison: actual zero5 versus selected prefix5, fixed probes.

Consumes externally SHA-bound MAIN strict receipts, not another physical job.
Elapsed commanded time is matched; state/contact mediation is NOT isolated.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_mixedmode_zero_time_control_20261028_v1 as zero
from experiments.spectral_reachability import agent_mixedmode_capability_analysis_20261028_v1 as math

NAME='agent_mixedmode_zero_signed_comparison_20261028_v1'
ROOT=zero.ROOT
probe,physical=zero.probe,zero.helper
require,sha,canonical=probe.require,probe.sha,probe.canonical
CAP,FLOOR=1024**2,200*1024**2
ZERO_REPORT_SHA='dfba2b92f1e4c7a4f56a8d8168fca427c0595236a21588aea1a97cae0f897e8e'
ZERO_RECEIPT_SHA='7c22af94f7aebaabbe4857cb1797c1b7fe857020145829273fb2349b45ea8e19'
SELECTED_REPORT_SHA=math.REPORT_SHA
SELECTED_RECEIPT_SHA='d1efceb6aa0eb4e72a586bb0c1ba297ecb433f12e048e7a0c966450b71156537'
PROBE_SHA='e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5'
ZERO_SOURCE_SHA='2f5cb42d0374a1d6d7d53b50fd9bfdd82bb502864795ac64341ac46bd67cb5aa'
ARMS=('zero5','selected_prefix5')
SPACES=math.SPACES


def contract():
    return dict(schema=NAME+'.contract',task_order=[[s,c] for s,c in probe.ORDER],
        arms=list(ARMS),probes=[k for k,_ in probe.PROBES],spaces=list(SPACES),
        elapsed_public_steps_before_probe=15,probe_steps=5,branches_per_arm=20,
        direction='original root-to-goal unit direction shared across both arms; angle wrapped separately',
        finite_response='endpoint(probe)-endpoint(zero) within same task/intervention',
        secant='(endpoint(+axis)-endpoint(-axis))/(2*original_float32_amplitude)',
        changes='selected_prefix5 minus zero5; no abs, winner, fitted Q, or sign-selected direction',
        receipt_authority='external SHA-bound completed MAIN strict receipts; do not rerun physical consume',
        time_matched=True,state_velocity_contact_mediation_isolated=False,
        known_roots=2,fresh_roots=0,P1_admission=False,independent_confirmation=False,equal_total_compute=False)


def accepted_receipts(zr,zw,sr,sw,zsha,ssha):
    """Two real distinct layouts, not a guessed generic receipt adapter."""
    require(zr['schema']==zero.NAME+'.report' and zr['status']=='complete' and zr['errors']==[] and
        zr['fixed_denominator']==len(zr['rows'])==20 and zr['contract']==zero.contract(), 'complete own zero20 report')
    require(zw['schema']==zero.NAME+'.consume' and zw['status']=='zero_time_control_responses_verified_not_admitted' and
        zw['fixed_denominator']==20 and zw['report_sha256']==zsha and
        zw['prepare_sha256']==zr['execution_binding']['prepare_sha256'], 'actual zero strict receipt binding')
    require(sr['schema']==probe.NAME+'.evidence' and sr['executor_schema']==physical.NAME+'.report' and
        sr['status']=='complete' and sr['errors']==[] and len(sr['rows'])==40 and
        sr['contract']==physical.contract(),'original complete40 physical ABI, not old post-only evidence')
    require(sw['schema']=='MAIN.mixedmode_capability_acceptance.v1','actual MAIN acceptance envelope')
    sc=sw['receipt']
    require(sc['schema']==probe.NAME+'.consume' and sc['status']=='saved_probe_responses_not_admitted' and
        sc['fixed_denominator']==40 and sc['executor_report_sha256']==sc['evidence_sha256']==ssha and
        sc['executor_prepare_sha256']==sr['executor_prepare_sha256'] and sc['prepare_sha256']==PROBE_SHA and
        sc['journal_sha256']==sr['journal']['sha256'] and
        sc['executor_source_sha256']==sr['executor_source_sha256'], 'original strict nested receipt exact binding')
    for c,r in ((zw,zr),(sc,sr)):
        require(c['rows']==r['rows'] and c['counts']==r['counts'] and c['errors']==r['errors']==[] and
            c['P1_admission'] is False and c['independent_confirmation'] is False and
            c['new_physics_calls']==c['new_neural_calls']==0,'accepted all rows/counts, no partial substitution')
    require(sw['P1_admission'] is False and sw['new_physics_calls']==sw['new_neural_calls']==0,
        'MAIN receipt is not scientific admission')
    return zw,sc


def matched_rows(p,zp,zr,sr):
    """Pure task/action matching seam. No starts are required equal across arms."""
    require(p['schema']==probe.NAME+'.prepare' and p['contract']==probe.contract() and
        len(p['tasks'])==4 and len(p['jobs'])==40 and zp['schema']==zero.NAME+'.prepare' and
        zp['contract']==zero.contract() and zp['probe_digest']==probe.digest(p) and
        zp['jobs']==zero.jobs_for(p) and len(zr['rows'])==20 and len(sr['rows'])==40,
        'same full original contexts and fixed intervention denominator')
    require(zr['execution_binding']['probe_prepare_sha256']==sr['execution_binding']['probe_prepare_sha256']==PROBE_SHA and
        sr['prepare_sha256']==PROBE_SHA,'same source tasks/probe prepare external SHA')
    out=[]
    for j,z in zip(zp['jobs'],zr['rows'],strict=True):
        parent=p['jobs'][j['task']*10+5+next(i for i,(k,_) in enumerate(probe.PROBES) if k==j['probe_id'])]
        s=sr['rows'][parent['index']];task=p['tasks'][j['task']];native=task['native']
        require(all(z[k]==v for k,v in j.items()) and parent['role']==s['role']=='after' and
            s['job_digest']==parent['job_digest']==z['parent_after_job_digest'] and
            s['context_digest']==z['context_digest']==task['context_digest'] and
            s['task']==z['task'] and s['probe_id']==z['probe_id'] and
            s['causal_prefix_commands']==parent['prefix_commands']==task['plan']['prefix_commands'] and
            parent['selection_digest']==z['parent_selection_digest']==task['plan']['selection_digest'],
            'root/task/probe/parent action ancestry join, not guessed row index')
        require(probe.raw(j['intervention_commands'],5).tobytes()==np.zeros((5,2),'<f4').tobytes() and
            probe.raw(j['probe_commands'],5).tobytes()==probe.raw(parent['probe_commands'],5).tobytes(),
            'literal zero5 intervention and same original probe amplitude/order')
        for r in (z,s):
            require(r['status']=='completed' and r['start_verified'] is True and r['error'] is None and
                r['attempted']==r['completed']==5 and r['condition']==task['plan']['condition'] and
                all(r[k]==native[k] for k in ('root_state','goal','root_sha256','goal_pose')) and
                r['goal_pixel_sha256']==native['goal_pixels'][r['condition']] and len(r['frames'])==6,
                'all matched full native7 root/goal and condition identities')
            for k in ('start_state','end_state'):probe.vector(r[k])
            require(r['frames'][0]['state']==r['start_state'] and r['frames'][0]['pre']==r['frames'][0]['post'] and
                r['frames'][0]['pre']['snapshot']['state']==r['start_state'] and
                r['frames'][0]['pre']['renderer']==native['renderer'] and
                r['frames'][0]['pre']['goal_pose']==native['goal_pose'],'true probe-start snapshot/renderer')
        require(z['zero5_executed'] is z['rgb_guard_accepted'] is True and z['model_selected_witness'] is False and
            s['causal_replay_verified'] is s['rgb_guard_accepted'] is True,'actual intervention/replay guard receipts')
        out.append((j,z,s))
    return out


def check_elapsed(pairings,z_events,s_events):
    """Check actual returned full warm10+prefix5+probe5 and resource axes per pair."""
    totals={a:{} for a in ARMS}
    for j,z,s in pairings:
        for arm,row,events in ((ARMS[0],z,z_events),(ARMS[1],s,s_events)):
            scoped=[e for e in events if e.get('job_digest')==row['job_digest']]
            raw=[e for e in scoped if e['kind']=='raw_return']
            prefix=j['intervention_commands'] if arm==ARMS[0] else s['causal_prefix_commands']
            expected=np.concatenate([zero.moving.warmup_commands(),probe.raw(prefix,5),probe.raw(j['probe_commands'],5)])
            require(len(raw)==20,'actual complete20, not nominal elapsed metadata')
            for i,(e,a) in enumerate(zip(raw,expected,strict=True)):
                phase='warmup' if i<10 else ('zero' if arm==ARMS[0] else 'prefix') if i<15 else 'probe'
                step=i+1 if i<15 else i-14
                require(e['phase']==phase and e['step']==step and
                    e['raw_bits']==e['actual_bits']==a.tobytes().hex() and e['actual_dtype']==np.dtype('<f4').str and
                    e['terminated'] is e['truncated'] is False, 'actual same elapsed/raw history, no native padding')
            require(raw[9]['state']==row['root_state'] and raw[14]['state']==row['start_state'] and
                raw[19]['state']==row['end_state'],'warm-root/intervention/probe actual labels join')
            counts={}
            for e in scoped:
                if e['kind'] not in ('operation_attempted','operation_returned'):continue
                c=counts.setdefault(e['resource'],dict(attempted=0,completed=0))
                c['attempted' if e['kind']=='operation_attempted' else 'completed']+=1
            expected_counts={('internal_physics_step' if k=='physics_step' else k):dict(attempted=v//20,completed=v//20)
                for k,v in zero.CAPS.items()}
            require(counts==expected_counts,'actual matched-time physical resource axes per independent branch')
            for k,c in counts.items():
                total=totals[arm].setdefault(k,dict(attempted=0,completed=0))
                for axis in ('attempted','completed'):total[axis]+=c[axis]
    return totals


def compare(p,pairings):
    """All four original spaces, all five probes. No outcome-dependent rules."""
    groups=[];comparisons=[]
    for ti,task in enumerate(p['tasks']):
        pairs=[x for x in pairings if x[0]['task']==ti]
        require([x[0]['probe_id'] for x in pairs]==[k for k,_ in probe.PROBES],'all ordered fixed five probes')
        native=task['native'];amplitude=float(probe.raw(pairs[1][0]['probe_commands'],5)[0,0])
        current=[]
        for ai,arm in enumerate(ARMS,1):
            rows=[x[ai] for x in pairs];start=probe.vector(rows[0]['start_state'])
            snapshot=rows[0]['frames'][0]['pre']['snapshot']
            require(all(probe.vector(r['start_state']).tobytes()==start.tobytes() and
                r['frames'][0]['pre']['snapshot']==snapshot for r in rows),'within-arm identical independent causal starts')
            endpoints={r['probe_id']:probe.vector(r['end_state']) for r in rows};z=endpoints['zero'];spaces={}
            for space in SPACES:
                u=math.task_direction(native,space);responses=[]
                for probe_id,_ in probe.PROBES:
                    d=math.difference(endpoints[probe_id],z,space)
                    responses.append(dict(probe_id=probe_id,zero_relative_delta=d.tolist(),
                        signed_original_task_projection=math.scalar_projection(d,u)))
                columns=[];asym=[];cuts=[]
                for axis in ('x','y'):
                    plus,minus=endpoints['+'+axis],endpoints['-'+axis]
                    columns.append((math.difference(plus,minus,space)/(2*amplitude)).tolist())
                    asym.append((math.difference(plus,z,space)+math.difference(minus,z,space)).tolist())
                    cuts.append(bool(abs(plus[4]-minus[4])>=np.pi) if space=='object_angle_wrapped' else None)
                spaces[space]=dict(original_task_direction=None if u is None else u.tolist(),
                    direction_status='zero_original_residual_no_direction' if u is None else 'fixed_original_root_to_goal',
                    finite_responses=responses,central_secant_columns=columns,
                    signed_original_task_secants=[math.scalar_projection(c,u) for c in columns],
                    asymmetry_columns=asym,angle_branch_cut=cuts,zero_drift=math.difference(z,start,space).tolist(),
                    secant_units='rad/raw_held_command' if space=='object_angle_wrapped' else 'px/raw_held_command',
                    unsigned_spectrum=None,full_task_success_inferred=False)
            group=dict(task=ti,root_slot=task['plan']['root_slot'],condition=task['plan']['condition'],arm=arm,
                task_id=task['plan']['task_id'],context_digest=task['context_digest'],
                parent_selection_digest=task['plan']['selection_digest'],root_state=native['root_state'],goal=native['goal'],
                start_state=start.tolist(),start_snapshot=deepcopy(snapshot),probe_amplitude_raw=amplitude,
                elapsed_public_steps_before_probe=15,endpoint_public_steps=20,
                evidence=[dict(probe_id=r['probe_id'],job_digest=r['job_digest'],end_state=r['end_state']) for r in rows],spaces=spaces)
            groups.append(group);current.append(group)
        z,s=current;spaces={}
        for space in SPACES:
            zs,ss=z['spaces'][space],s['spaces'][space]
            spaces[space]=dict(signed_response_changes=[dict(probe_id=x['probe_id'],selected_minus_zero=math.subtract(
                x['signed_original_task_projection'],y['signed_original_task_projection']))
                for x,y in zip(ss['finite_responses'],zs['finite_responses'],strict=True)],
                signed_secant_changes=[math.subtract(x,y) for x,y in zip(ss['signed_original_task_secants'],zs['signed_original_task_secants'],strict=True)])
        comparisons.append(dict(task=ti,root_slot=s['root_slot'],condition=s['condition'],spaces=spaces,
            elapsed_matched=True,state_matched=z['start_state']==s['start_state'],
            snapshot_matched=z['start_snapshot']==s['start_snapshot'],state_mediation_isolated=False,
            causal_capability_gain='unknown',P1_admission=False))
    return dict(schema=NAME+'.report',status='matched_time_signed_responses_described_not_admitted',contract=contract(),
        denominators=dict(known_roots=2,root_conditions=4,arms=2,matched_branches=40,probes_per_arm=20,
            signed_projections=160,signed_secants=64,response_changes=80,secant_changes=32,fresh_roots=0),
        groups=groups,comparisons=comparisons,new_neural_calls=0,new_physics_calls=0,P1_admission=False,
        independent_confirmation=False,equal_total_compute=False,
        limitations=['elapsed commanded time matched, not state/velocity/contact/cache mediation isolation',
            'original shared root-to-goal direction, never refit to post-prefix residual',
            'jointXY includes agent and object; object-only response is not complete task success or failure',
            'angle wrapped separately, no fitted pixel/radian weighting',
            'five constant XY controls at the original single amplitude, not full ten-coordinate derivatives',
            'signed secants are local finite responses, not terminal-distance gain or causal ability proof',
            'terminal snapshot does not establish whole-trajectory contact',
            'known-root description; no fresh-root claim, P1 admission, or theoretical rejection'])


def assemble(zero_output,zero_receipt,selected_output,selected_receipt,
             zero_report_sha=ZERO_REPORT_SHA,zero_receipt_sha=ZERO_RECEIPT_SHA,
             selected_report_sha=SELECTED_REPORT_SHA,selected_receipt_sha=SELECTED_RECEIPT_SHA):
    expected=(ZERO_REPORT_SHA,ZERO_RECEIPT_SHA,SELECTED_REPORT_SHA,SELECTED_RECEIPT_SHA)
    require((zero_report_sha,zero_receipt_sha,selected_report_sha,selected_receipt_sha)==expected,'fixed actual pair only')
    bindings={}
    def read(path,h):
        path=Path(path).resolve();require(sha(path)==probe.hash_value(h),'external actual input SHA')
        bindings[str(path)]=h;return probe.read(path)
    zd,sd=Path(zero_output).resolve(),Path(selected_output).resolve()
    zr=read(zd/'report.json',zero_report_sha);zw=read(zero_receipt,zero_receipt_sha)
    sr=read(sd/'report.json',selected_report_sha);sw=read(selected_receipt,selected_receipt_sha)
    accepted_receipts(zr,zw,sr,sw,zero_report_sha,selected_report_sha)
    require(sha(zero.__file__)==ZERO_SOURCE_SHA and sha(physical.__file__)==math.PHYSICAL_SOURCE_SHA and
        zr['executor_source_sha256']==ZERO_SOURCE_SHA,'reviewed frozen executor sources')
    zp=read(zr['execution_binding']['prepared'],zr['execution_binding']['prepare_sha256'])
    sp=read(sr['execution_binding']['prepared'],sr['executor_prepare_sha256'])
    require(sp['schema']==physical.NAME+'.prepare' and sp['contract']==physical.contract() and
        sp['probe_digest']==zp['probe_digest'] and
        zp['probe_prepare_sha256']==sp['probe_prepare_sha256']==PROBE_SHA,'common original source/prepare closure')
    p=read(zp['probe_prepared'],PROBE_SHA)
    require(sha(sp['probe_prepared'])==PROBE_SHA and sha(sr['execution_binding']['probe_prepared'])==PROBE_SHA,
        'selected recipe uses same exact original probe prepare')
    pairings=matched_rows(p,zp,zr,sr);events=[]
    for directory,r in ((zd,zr),(sd,sr)):
        require(r['journal']['file']=='journal.jsonl' and sha(directory/'journal.jsonl')==r['journal']['sha256'],
            'actual accepted journal SHA')
        bindings[str(directory/'journal.jsonl')]=r['journal']['sha256']
        events.append(physical.read_events(directory/'journal.jsonl',r['journal_tail']))
    budgets=check_elapsed(pairings,*events);result=compare(p,pairings)
    result.update(input_file_sha256=bindings,accepted_strict_receipt_authority=True,
        full_producer_consume_reexecuted=False,matched_branch_resources=budgets,
        original_run_resources=deepcopy(sr['counts']),zero_run_resources=deepcopy(zr['counts']),
        original_run_denominator=40,original_before_branches_excluded_by_preregistered_role=20,
        parent_model_search_cost='inherited mixedmode ancestry, not new compute-equivalence claim',
        own_file_sha256={str(f.resolve()):sha(f) for f in [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),Path(math.__file__)]})
    for path,h in bindings.items():require(sha(path)==h,'inputs unchanged during saved comparison')
    require(len(canonical(result))<=CAP,'bounded1MiB comparison')
    return result


def write_new(path,result):
    path=Path(path).absolute();raw=canonical(result)
    require(path.name.startswith(NAME+'_') and path.suffix=='.json' and len(raw)<=CAP and not path.is_symlink() and
        shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive own output1MiB and floor200MiB')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'complete output write');f.flush();os.fsync(f.fileno())
    zero.adapter.paired.fsync_dir(path.parent)
    return dict(output=str(path),sha256=sha(path),bytes=len(raw))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero-output',default=str(ROOT/'results'/ (zero.NAME+'_MAIN01')))
    parser.add_argument('--zero-receipt',default=str(ROOT/'results/consume_zero_and_rolling_MAIN_20261028_v1_zero_MAIN01.json'))
    parser.add_argument('--selected-output',default=str(ROOT/'results'/ (physical.NAME+'_MAIN01')))
    parser.add_argument('--selected-receipt',default=str(ROOT/'results/consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json'))
    for k,d in (('zero-report-sha',ZERO_REPORT_SHA),('zero-receipt-sha',ZERO_RECEIPT_SHA),
                ('selected-report-sha',SELECTED_REPORT_SHA),('selected-receipt-sha',SELECTED_RECEIPT_SHA)):
        parser.add_argument('--'+k,default=d)
    parser.add_argument('--destination')
    a=vars(parser.parse_args(argv));destination=a.pop('destination');result=assemble(**a)
    print(canonical(result if destination is None else write_new(destination,result)).decode())


if __name__=='__main__':main()
