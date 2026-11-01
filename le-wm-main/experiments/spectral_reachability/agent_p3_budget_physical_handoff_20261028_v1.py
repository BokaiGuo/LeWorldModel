"""Saved-only exact physical handoff from MAIN's full budget HOST receipt.

No host.consume, neural/physics import, execution entry, or future SHA defaults.
This authenticates persisted MAIN authority; it is NOT a replacement host audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_child_budget_host_cuda_20261028_v1 as host
from experiments.spectral_reachability import agent_p3_child_native_physical_saved_20261028_v2 as saved

NAME='agent_p3_budget_physical_handoff_20261028_v1'
ROOT,core,contract=host.ROOT,host.core,host.contract
physical=saved.base
require,sha,digest,canonical=host.require,host.sha,host.digest,host.canonical
READ=host.base.read
FLAGS=dict(host.FLAGS,compute_advantage=False,independent_confirmation=False,
           eqcompute=False,freshness=False)
PINS=dict(zip(map(str,host.own_files()),(
    'a0dbd2bcae92fc10e541500ee3b8247ec72a7e5c7a79c67f0248761174354b14',
    '38226f6100a24a8365de60f8386562faee91f3d79d0a96c783f943fcd953f528',
    'eedb3a2974fb07d68176c45db3936999e792cb6386ca93948244a888fc46dec3'),strict=True))
PINS.update(dict(zip(map(str,saved.own_files()),(
    '9c6fe36958042988fce53f12fca93476c331f91eca9e63bafebf87df4e1042cc',
    '9610dffc9887a42701e7a0bea6c66d7a7c5aa9d5ec4d115e90ed7882f6ebd542',
    'fecbbb38c2bb126db7f8f7df0be53ff4c20418824ea45f2adeda7ec3c1a69bf0'),strict=True)))


def own_files():
    return [Path(__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']


def inputs(report,report_sha,receipt,receipt_sha,reuse_policy):
    require(reuse_policy in ('strict_saved','none'),'explicit reuse policy, never result-driven fallback')
    for h in (report_sha,receipt_sha):core.hash_value(h)
    return dict(host_report=str(contract.absolute(report)),report_sha256=report_sha,
        strict_receipt=str(contract.absolute(receipt)),receipt_sha256=receipt_sha,reuse_policy=reuse_policy)


def expected_receipt(r,report_sha):
    """Match the entire frozen consumer ABI, not a CLI summary.

    The external SHA provided by MAIN is the authority for completed fullarray
    verification. Reconstructing this metadata alone is not scientific evidence.
    """
    rows=r['session']['rows'];census=dict(compiled=0,refused=0,not_reached=0)
    for row in rows:
        if row['status']=='not_reached':census['not_reached']+=1
        elif row['status']=='budget_complete_native_unverified':census['compiled']+=1
        else:census['refused']+=1
    actual=r['session']['gate']['actual']
    return dict(schema=host.NAME+'.consume',status='interleaved_native_fullarrays_budget_verified_not_admitted',
        report_sha256=report_sha,prepare_sha256=r['prepare_sha256'],fixed_denominator=48,child_denominator=24,
        independent_roots=2,census=census,actual_resources=actual,same_cap=True,
        actual_vectors_equal=actual['fixed']==actual['adaptive'],FLOPs=None,full_rows=rows,
        gate_receipt_sha256=digest(r['session']['gate']),
        verification_scope='saved full arrays, original chooser/ridge/replay and gate events; no neural/AD recomputation or GPU authenticity proof',
        saved_verification=dict(signal_array_SVD_calls=2*(census['compiled']+census['refused']),
            compiler_array_SVD_calls=census['compiled'],outside_native_ledger=True,neural_calls=0,physics_calls=0),**host.FLAGS)


def validate_host_link(p,r,receipt,report_sha):
    require(p['schema']==host.NAME+'.prepare' and p['profile']==host.profile() and
        len(p['wiring']['children'])==24 and len(p['wiring']['tasks'])==2,'actual bound host24 schema/profile')
    require(r['schema']==host.NAME+'.report' and r['status']=='complete_native_panel_not_admitted' and
        r['errors']==[] and r['sources_verified'] is r['cleanup_ok'] is True and
        r['source_manifest_sha256']==digest(p['sources']) and
        all(r[k] is p[k] is False for k in host.FLAGS),'complete closed-gate report required BEFORE physical handoff')
    session=r['session']
    require(len(session['rows'])==48 and session['fixed_denominator']==48 and session['child_denominator']==24 and
        session['stopped'] is False and session['gate']['failed'] is False and session['gate']['cursor']==48 and
        not session['gate']['pending'],'all48 terminal paid gate, not a successful partial subset')
    require(canonical(receipt)==canonical(expected_receipt(r,report_sha)),
            'MAIN persisted COMPLETE full strict receipt exact ABI/report/rows/gate/actual binding; no summary')
    for i,row in enumerate(session['rows']):
        child=contract.validate_child(p['wiring']['children'][i//2],p['wiring']['tasks']);mode=contract.MODES[i%2]
        require((row['child_id'],row['mode'])==(child['child_id'],mode),'fixed same24child48slot order')
        require(row['error'] is None,'failed engineering slot cannot be promoted')
        if child['status']=='not_reached':
            require(row['status']=='not_reached' and all(row[k] is None for k in
                ('signal','signal_resources','decision','compiler')),'not_reached remains unknown')
            continue
        require(row['signal'] is not None and row['decision'] is not None and row['signal_resources'] is not None,
                'refusal must retain actually paid signal and original decision')
        task=p['wiring']['tasks'][child['root_slot']]
        signals=core.FrozenSignals(**row['signal']['scheduling_signal'])
        require(row['decision']==core.choose(task,dict(context=child['context'],depth=0),signals,mode),
                'unchanged frozen chooser; no post-outcome reallocation')
        require(row['signal_resources']['attempted']==row['signal_resources']['completed']==contract.signal_caps(1),
                'independent whole actual paid signal, no shared/borrowed root J')
        key=child['child_id']+':'+mode;node=session['gate']['nodes'][key]
        if row['status']=='budget_complete_native_unverified':
            c=row['compiler']
            require(node['status']==row['status'] and c is not None and c['status']==host.compiler.FINAL and
                c['error'] is None and c['selected'] is not None and
                c['selected']['parent_context_sha256']==child['context_sha256'] and
                c['branch_sha256']==digest(host.branch_for(child,mode,row)),'only full same-child strict compiled selected prefix')
            physical.raw(c['selected']['raw5'],5)
        else:
            require(row['compiler'] is None and node['status']=='refused_after_paid_signal' and
                row['status'] in ('original_whole_node_search_budget','unsupported_reconfiguration','original_policy_stopped'),
                'original paid whole-node refusal, no zero/fabricated command')


def load_host(authority):
    # Check ALL external boundaries before any saved upstream reconstruction.
    report=contract.absolute(authority['host_report']);receipt_path=contract.absolute(authority['strict_receipt'])
    require(report.is_file() and receipt_path.is_file(),'missing terminal report or MAIN complete strict receipt: no future SHA authority')
    require(sha(report)==authority['report_sha256'] and sha(receipt_path)==authority['receipt_sha256'],
            'external actual host report and complete strict SHA')
    r=READ(report,256*1024**2);receipt=READ(receipt_path,256*1024**2)
    path=contract.absolute(r['prepared']);require(sha(path)==r['prepare_sha256'],'actual HOST prepare edge')
    p=READ(path,64*1024**2);contract.check_sources(p['sources']);contract.check_sources(PINS)
    require(all(p['sources'].get(k)==v for k,v in PINS.items() if k in set(map(str,host.own_files()))),
            'HOST actual prepare pins reviewed consumer source/tests/doc')
    validate_host_link(p,r,receipt,authority['report_sha256'])
    require(sha(report)==authority['report_sha256'] and sha(receipt_path)==authority['receipt_sha256'],
            'host report/strict unchanged after saved handoff checks')
    return p,r,receipt


def load_physical(reuse_policy):
    saved.pins();contract.check_sources(PINS)
    require(sha(saved.ACTUAL_OUTPUT/'report.json')==saved.ACTUAL_REPORT_SHA,'sealed original physical report SHA')
    r=READ(saved.ACTUAL_OUTPUT/'report.json',physical.FINAL)
    require(r['prepare_sha256']==saved.ACTUAL_PREPARE_SHA and sha(r['prepared'])==saved.ACTUAL_PREPARE_SHA,
            'sealed original native restoration prepare')
    p=physical.verify(r['prepared'],saved.ACTUAL_PREPARE_SHA)  # saved only; includes P4 full native source checks
    strict=None
    if reuse_policy=='strict_saved':
        strict=saved.consume(str(saved.ACTUAL_OUTPUT),saved.ACTUAL_REPORT_SHA)
        require(strict['complete'] is True and strict['status']=='native_children_verified_not_admitted' and
            canonical(strict['rows'])==canonical(r['rows']),'full old physical.consume required before ANY evidence reuse')
    return p,r,strict


def raw_identity(value,count):
    a=physical.raw(value,count)
    header=dict(dtype=a.dtype.str,shape=list(a.shape))
    return dict(**header,bytes_hex=a.tobytes().hex(),sha256=hashlib.sha256(canonical(header)+b'\0'+a.tobytes()).hexdigest())


def start_identity(plan,manifest_sha):
    child=plan['child'];frames=plan['source_frames']
    require(child['status']=='observed_child' and frames and len(frames)==12+child['boundary'],
            'whole original warmup/prefix native restoration frames')
    for f in frames:physical.native_frame(f)
    require(frames[-1]['file_sha256']==child['observation']['png_sha256'] and
        frames[-1]['pixel_sha256']==child['observation']['pixel_sha256'],'same native PNG to encoded child context')
    return dict(physical_source_report_sha256=child['ancestry']['source_report_sha256'],
        source_manifest_sha256=manifest_sha,child=deepcopy(child),
        prefix_raw=raw_identity(child['executed_prefix'],child['boundary']),
        history_raw=raw_identity(child['raw_history10'],10),
        # Includes ALL snapshots/body float64 bits, goal, pose, image hashes and
        # source action ordinals, not merely seven wrapped state numbers.
        whole_source_frames=deepcopy(frames),source_returns=deepcopy(plan['source_returns']),
        context_sha256=child['context_sha256'],task_binding_sha256=child['ancestry']['task_binding_sha256'],
        goal_sha256=digest(frames[-1]['goal']),goal_rgb_sha256=frames[-1]['goal_rgb_sha256'])


def new_budget(rows):
    pending=[r for r in rows if r['route']=='pending_new_physics'];n=len(pending)
    prefix=sum(r['runner_plan']['child']['boundary'] for r in pending);public=15*n+prefix
    return dict(execute_slots=n,recovery_public_cap=10*n+prefix,own_public_cap=5*n,
        caps=dict(constructor=n,close=n,reset=2*n,set_state=4*n,goal_render=2*n,goal_setter=3*n,
            goal_bind=n,public_step=public,physics_step=4*n+10*public,explicit_render=17*n+prefix),
        native_earlystop_no_padding=True,nested_resources_not_added=True,new_actual_resources=None,
        historical993public_is_not_new_payment=True)


def project(p,r,oldp,oldr,oldstrict,reuse_policy,manifest_sha):
    require(len(oldp['rows'])==len(oldr['rows'])==48,'original native physical48 denominator')
    require(reuse_policy!='strict_saved' or (oldstrict is not None and oldstrict['complete'] is True and
        oldstrict['rows']==oldr['rows']),'strict old consumer required BEFORE choosing any reuse')
    rows=[]
    for i,hostrow in enumerate(r['session']['rows']):
        child=p['wiring']['children'][i//2];mode=contract.MODES[i%2];template=oldp['rows'][i]
        require(canonical(template['child'])==canonical(child) and template['mode']==mode and
            child['ancestry']['source_report_sha256']==physical.P4_SHA,'exact original same-slot native ancestry/context, never observed4')
        plan=deepcopy(template);plan['child']=deepcopy(child);plan['selected']=None
        row=dict(ordinal=i,child_id=child['child_id'],mode=mode,route='not_reached',
            paid_status=hostrow['status'],decision=deepcopy(hostrow['decision']),runner_plan=plan,
            start_sha256=None,command=None,goal_sha256=None,selection_digest=None,reuse_evidence=None,
            freshness=False,new_actual_resources=None)
        if child['status']=='not_reached':
            require(hostrow['status']=='not_reached','original unknown pair retained');plan['route']='not_reached'
        else:
            identity=start_identity(template,manifest_sha);row['start_sha256']=digest(identity)
            row['goal_sha256']=identity['goal_sha256']
            if hostrow['status']!='budget_complete_native_unverified':
                require(hostrow['compiler'] is None,'paid refused cannot issue raw5');row['route']='paid_refused'
                plan['route']=hostrow['status']
            else:
                selected=hostrow['compiler']['selected'];plan['selected']=deepcopy(selected)
                row.update(route='pending_new_physics',command=raw_identity(selected['raw5'],5),
                    selection_digest=selected['selection_digest']);plan['route']='execute'
                # Decide ONLY on exact source/start/actions, before reading any
                # old reward/success/earlystop field. No cross-slot trajectory search.
                same_command=template['selected'] is not None and row['command']==raw_identity(template['selected']['raw5'],5)
                if reuse_policy=='strict_saved' and same_command:
                    historical=oldstrict['rows'][i]
                    require(historical['ordinal']==i and historical['plan_sha256']==digest(template) and
                        historical['status'] in ('returned5','native_stopped') and historical['error'] is None,
                        'strict complete same-slot old branch')
                    physical.same_frame(historical['boundary'],template['source_frames'][-1])
                    require(digest(start_identity(plan,manifest_sha))==row['start_sha256'],
                            'whole source/fullstate bits/goal/context equality before reuse')
                    row['route']='reused_saved';plan['route']='reused_saved'
                    row['reuse_evidence']=dict(report_sha256=saved.ACTUAL_REPORT_SHA,prepare_sha256=saved.ACTUAL_PREPARE_SHA,
                        original_ordinal=i,original_plan_sha256=digest(template),full_old_row=deepcopy(historical),
                        start_sha256=row['start_sha256'],command_sha256=row['command']['sha256'],
                        strict_consumer=saved.NAME,fresh_execution=False,
                        identity_only_selection=True,new_physics_calls=0)
        rows.append(row)
    census={k:sum(row['route']==k for row in rows) for k in ('pending_new_physics','reused_saved','paid_refused','not_reached')}
    require(sum(census.values())==48,'all48 handoff routes without selection/padding')
    return dict(rows=rows,census=census,new_budget=new_budget(rows),fixed_denominator=48,child_denominator=24,
        independent_roots=2,new_neural_calls=0,physical_execution_started=False,
        runner_entry=dict(module=physical.NAME,function='execute',schema=NAME+'.runner_plan',
            accepts='runner_plan of pending_new_physics slots only; original refs/source_plan from source_manifest',
            old_CLI_prepare_compatible=False,requires='MAIN reviewed authorized runner and saved consumer for this NEW handoff schema'),
        historical_resources=None if oldstrict is None else dict(resources=oldstrict['resources'],
            scope='old full panel including repeated trajectories, NOT new fees or independent confirmations'),**FLAGS)


def material(authority):
    contract.check_sources(PINS)
    p,r,receipt=load_host(authority)  # intentionally NO host.consume invocation
    oldp,oldr,oldstrict=load_physical(authority['reuse_policy'])
    manifest=contract.absolute(oldp['source_manifest']);manifest_sha=sha(manifest)
    require(p['sources'].get(str(manifest))==manifest_sha and
        p['sources'].get(str(physical.P4_OUTPUT/'report.json'))==physical.P4_SHA,
        'same actual task/seed/placement/warmup/source manifest bound by HOST and native source')
    out=project(p,r,oldp,oldr,oldstrict,authority['reuse_policy'],manifest_sha)
    sources=deepcopy(p['sources']);contract.merge(sources,oldp['sources']);contract.merge(sources,PINS)
    contract.merge(sources,{str(f):sha(f) for f in own_files()})
    extra=[Path(authority['host_report']),Path(authority['strict_receipt']),Path(r['prepared']),
        saved.ACTUAL_OUTPUT/'report.json',Path(oldr['prepared']),manifest]
    if oldstrict is not None:
        extra.extend([saved.ACTUAL_OUTPUT/'journal.jsonl',saved.ACTUAL_OUTPUT/'binding.json'])
        for f in saved.ACTUAL_OUTPUT.iterdir():
            if f.suffix=='.png':extra.append(f)
    contract.merge(sources,{str(f):sha(f) for f in extra});contract.check_sources(sources)
    require(sha(authority['host_report'])==authority['report_sha256'] and
        sha(authority['strict_receipt'])==authority['receipt_sha256'],'final host/strict boundary')
    return dict(schema=NAME+'.prepare',authority=deepcopy(authority),source_manifest=str(manifest),
        sources=sources,host_actual_resources=receipt['actual_resources'],handoff=out,
        full_MAIN_strict_receipt_required=True,host_consume_reexecuted=False,physical_dispatch_allowed=False,**FLAGS)


def verify(prepared,expected):
    prepared=contract.absolute(prepared);core.hash_value(expected)
    require(sha(prepared)==expected,'external actual handoff prepare SHA')
    p=READ(prepared,64*1024**2);contract.check_sources(p['sources'])
    require(canonical(p)==canonical(material(p['authority'])) and sha(prepared)==expected,
            'complete saved authority/routes/source/raw bits reconstruction')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare')
    for flag in ('host-report','report-sha256','strict-receipt','strict-sha256','output'):a.add_argument('--'+flag,required=True)
    a.add_argument('--reuse-policy',choices=('strict_saved','none'),required=True)
    a=sub.add_parser('verify');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha256',required=True)
    args=parser.parse_args()
    if args.cmd=='prepare':
        output=contract.absolute(args.output)
        require(output.parent==ROOT/'results' and output.name.startswith(NAME+'_') and not output.exists(),
                'NEW own output before any saved consumers')
        authority=inputs(args.host_report,args.report_sha256,args.strict_receipt,args.strict_sha256,args.reuse_policy)
        result=physical.write_new(output,material(authority))
    else:
        p=verify(args.prepared,args.prepare_sha256)
        result=dict(status='saved_physical_handoff_verified_NOT_executed',census=p['handoff']['census'],**FLAGS)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
