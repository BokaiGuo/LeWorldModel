"""MAIN-only budget-gated CUDA lifecycle and streaming fullarray consumer.

prepare/verify/consume import no torch and execute no neural/physical backend.
All48 slots remain; equal resource ceilings are not matched actual compute.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_p3_child_budget_native_20261028_v1 as native

NAME = 'agent_p3_child_budget_host_cuda_20261028_v1'
ROOT, core, contract = native.ROOT, native.core, native.contract
budget, compiler, paid = native.budget, native.compiler, native.paid
base = compiler.base
require, sha, digest, canonical = native.require, native.sha, native.digest, native.canonical
FLAGS = dict(budget.FLAGS)
UPSTREAM = ROOT/'results/agent_p3_child_budget_native_20261028_v1_prepare_MAIN01.json'
UPSTREAM_SHA = '22f162fb493b6b83b30bddd66ec1b7f900f4bf36ea73c6249a5a97258002e50d'
NATIVE_SHAS = ('9ffc6a9c85ca077192fa686e8a80a48cc45a9bfddd86238f4aaf0bc132b20f6a',
    '5b71eaaf66f6fbfb52c29d4be1c08b8b618f8414d262474119809c7f24a39121',
    '74ed62e47511f222838ddd0552a73729e6981b7c67a0b8fe3bc51fd728239029')
AUTH = 'MAIN_APPROVED_BUDGET_CHILD_CUDA'


def profile():
    return dict(output_bytes=512*1024**2, final_reserve_bytes=256*1024**2,
        disk_floor_bytes=200*1024**2, wall_seconds=1800,
        cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        decoded_journal_bytes=paid.DECODED_CAP, journal_line_bytes=paid.LINE_CAP,
        slot_trace_bytes=64*1024**2, saved_receipt_bytes=256*1024**2,
        TF32=False, SDP='MATH', CPU_neural=False, no_retry=True,
        wall_scope='before verify through synchronize/model/source guards and journal close; terminal report write excluded',
        capacity_forecast_is_not_a_mathematical_bound=True)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def pins():
    result = dict(zip(map(str,native.own_files()),NATIVE_SHAS,strict=True))
    contract.check_sources(result)
    return result


def material(prepared, expected):
    pins()
    require(str(contract.absolute(prepared))==str(UPSTREAM) and expected==UPSTREAM_SHA,
            'fixed actual native prepare path/SHA; no future SHA or cohort replacement')
    p = native.verify(prepared, expected)  # mandatory full producer/source closure
    report = base.read(budget.REPORT,256*1024**2)
    original_path = contract.absolute(report['prepared'])
    require(sha(original_path)==report['prepare_sha256'], 'actual original compiler prepare')
    original = base.read(original_path,64*1024**2)
    require(original['children']==p['children'] and original['tasks']==p['tasks'],
            'same original24 child/context/tasks, no substituted policy')
    sources = deepcopy(p['sources'])
    contract.merge(sources,pins())
    contract.merge(sources,{str(f):sha(f) for f in own_files()})
    contract.merge(sources,{str(UPSTREAM):expected,str(original_path):report['prepare_sha256']})
    contract.check_sources(sources)
    return dict(schema=NAME+'.prepare',native_prepared=str(prepared),native_sha256=expected,
        wiring=p,policy=original['policy'],expected_model_hash=original['expected_model_hash'],
        sources=sources,profile=profile(),fixed_denominator=48,child_denominator=24,
        independent_roots=2,signal_sharing='independent',scientific_equal_actual=False,**FLAGS)


def verify(prepared, expected):
    path=contract.absolute(prepared); core.hash_value(expected)
    require(sha(path)==expected,'external actual host prepare SHA')
    p=base.read(path,64*1024**2); contract.check_sources(p['sources'])
    require(canonical(p)==canonical(material(p['native_prepared'],p['native_sha256'])) and sha(path)==expected,
            'complete native/full strict/source/profile rebuild')
    return p


def initial_rows(p):
    return [dict(child_id=c['child_id'],mode=mode,
        status='not_reached' if c['status']=='not_reached' else 'not_started',
        signal=None,signal_resources=None,decision=None,compiler=None,error=None)
        for c in p['wiring']['children'] for mode in contract.MODES]


def branch_for(child, mode, row):
    decision=row['decision']
    response=next(r for r in row['signal']['responses'] if r['horizon']==decision['horizon'])
    return dict(child_id=child['child_id'],root_slot=child['root_slot'],mode=mode,
        child_sha256=digest(child),parent_context_sha256=child['context_sha256'],status=compiler.READY,
        decision=deepcopy(decision),response=deepcopy(response))


def validate_slot(p, index, row, signal_trace, program_trace, gate, seal):
    """Only actual full native arrays can close an observed slot, never caps."""
    initial=initial_rows(p)[index]
    core.keys(row,tuple(initial))
    child=p['wiring']['children'][index//2]; mode=contract.MODES[index%2]
    require((row['child_id'],row['mode'])==(child['child_id'],mode),'same ordered slot')
    key=child['child_id']+':'+mode
    if child['status']=='not_reached':
        require(row==initial and not signal_trace and not program_trace and seal is None and key not in gate.nodes,
                'not_reached remains unknown without fabricated payment')
        return 'not_reached'
    task=p['wiring']['tasks'][child['root_slot']]
    require(row['error'] is None and row['signal'] is not None and row['signal_resources'] is not None,
            'complete observed paid signal, not failed/partial metadata')
    responses=row['signal']['responses']
    require([r['horizon'] for r in responses]==[1,3] and all(
        r['producer_sha256']==sha(paid.__file__) and r['binding_sha256']==digest(task['binding']) and
        r['parent_context_sha256']==child['context_sha256'] for r in responses),
        'actual same-child independent source/goal/model/H1/H3, never root or sibling J')
    acquisition=dict(group=key,status='returned',error=None,backend_kind='HOST_CUDA',
                     signal=row['signal'],resource_receipt=row['signal_resources'])
    signals=paid.validate_trace(task,child,acquisition,signal_trace,1)
    schedule=[e for e in row['signal_resources']['events'] if e['resource']=='schedule']
    require(len(schedule)==1 and schedule[0]['scope']==dict(child_id=child['child_id'],group=mode,
        mode=mode,phase='schedule',parent_context_sha256=child['context_sha256']),
        'one actual paid original method schedule')
    decision=core.choose(task,dict(context=child['context'],depth=0),signals,mode)
    require(row['decision']==decision and seal is not None and seal['decision']==decision and
        seal['signals']==row['signal']['scheduling_signal'],'unchanged chooser and paid seal identity')
    node=gate.nodes[key]
    if seal['result']['dispatch_allowed']:
        require(row['status']=='budget_complete_native_unverified' and node['status']==row['status'] and
                row['compiler'] is not None,'full same-node compiler and gate finish required')
        branch=branch_for(child,mode,row)
        compiler.validate_branch(task,child,branch,row['compiler'],program_trace)
        return 'compiled'
    require(row['status']==seal['result']['reason'] and row['compiler'] is None and not program_trace and
            node['status']=='refused_after_paid_signal','whole-node paid refusal, no compile/subset/padding')
    return 'refused'


def replay_gate_event(gate, event):
    check=dict(event); h=check.pop('event_sha256')
    require(event['index']==len(gate.events) and event['previous_sha256']==(
        gate.events[-1]['event_sha256'] if gate.events else None) and digest(check)==h,'exact gate chain')
    kind=event['kind']
    if kind in ('opened','open_refused'): gate.open(**event['request'])
    elif kind=='attempted':
        e=event['entry'];gate.begin(e['ticket'],e['resource'],shape=e['shape'],directions=e['directions'])
    elif kind=='returned':gate.returned(event['operation_id'])
    elif kind=='sealed':gate.seal(event['ticket'],event['decision'],event['signals'])
    elif kind=='finished':gate.finish(event['ticket'])
    else:raise ValueError('failed/unknown gate event cannot certify a complete panel')
    require(gate.events[-1]==event,'exact original v2 event replay, no hidden edits')


def validate_stream(events, p, report):
    """Streaming one slot's arrays; online reconstruction of every gate event.

    A gate attempt MUST immediately precede its native attempt, and native
    return MUST immediately precede gate completion. No delayed preauthorization.
    """
    wiring=p['wiring'];gate=budget.WholeNodeGate(wiring['limits'],wiring['children'],wiring['tasks'])
    position=0;stage='denominator';phase='signal';signal_trace=[];program_trace=[]
    maps={};expect=None;seal=None;trace_bytes=0;tail='0'*64
    census=dict(compiled=0,refused=0,not_reached=0)
    for event,tail in events:
        kind=event['kind']
        if stage=='denominator':
            require(event==dict(kind='denominator',rows=initial_rows(p),fixed_denominator=48,child_denominator=24,
                sources_sha256=digest(p['sources'])),'full48 durable before any work')
            stage='before';continue
        if stage=='before':
            require(event==dict(kind='model_before',hashes=report['model_before']),'model guard before all work')
            stage='work';continue
        if kind=='work':
            core.keys(event,('kind','slot','payload'))
            require(stage=='work' and type(event['slot']) is int and event['slot']==position and position<48,
                    'exact contiguous fixed48 slot trace')
            e=event['payload']
            child=wiring['children'][position//2];mode=contract.MODES[position%2];key=child['child_id']+':'+mode
            if e['kind']=='budget_gate':
                core.keys(e,('kind','payload'));g=e['payload']
                if expect is not None:
                    require(expect[0]=='gate_return' and g['kind']=='returned' and
                            g['operation_id']==expect[1],'native return immediately before gate completion')
                    expect=None
                else:require(g['kind']!='returned','gate completion cannot precede actual return')
                replay_gate_event(gate,g)
                require(gate.cursor==position+1,'gate belongs to this exact original slot')
                if g['kind']=='attempted':
                    require(g['entry']['ticket']==key and g['entry']['phase']==phase,'same method/stage gate attempt')
                    expect=('native_attempt',g['entry'])
                elif g['kind']=='sealed':
                    require(phase=='signal' and seal is None and g['ticket']==key,'one signal seal before compile')
                    seal=g;phase='compile';maps={}
                elif g['kind']=='finished':require(g['ticket']==key,'same whole node gate finish')
            else:
                require(gate.cursor==position+1 and key in gate.nodes,'no native call before reserved same-child gate')
                if expect is not None:
                    require(expect[0]=='native_attempt' and e['kind']=='resource_attempt',
                            'gate predispatch must immediately precede actual native entry')
                    a=expect[1];n=e['entry']
                    require(all(a[k]==n[k] for k in ('resource','shape','directions','delta')) and
                            n['index'] not in maps,'gate/native actual operation bijection')
                    maps[n['index']]=a['operation_id'];expect=None
                else:require(e['kind']!='resource_attempt','ungated actual entry refused')
                if e['kind']=='resource_return':
                    require(e['entry']['index'] in maps,'actual return has preauthorized operation')
                    op=maps[e['entry']['index']]
                    require(op in gate.pending,'no duplicate/unattempted native return')
                    expect=('gate_return',op)
                trace_bytes+=len(canonical(e))
                require(trace_bytes<=p['profile']['slot_trace_bytes'],'bounded one-slot fullarray consumer')
                (signal_trace if phase=='signal' else program_trace).append(e)
        elif kind=='slot_return':
            core.keys(event,('kind','slot','row'))
            require(stage=='work' and position<48 and type(event['slot']) is int and event['slot']==position and
                    expect is None and gate.cursor==position+1,'all actual operations closed before exact slot return')
            row=report['session']['rows'][position]
            require(event['row']==row,'durable complete row before advancing method')
            result=validate_slot(p,position,row,signal_trace,program_trace,gate,seal)
            census[result]+=1;position+=1;phase='signal';signal_trace=[];program_trace=[];maps={};seal=None;trace_bytes=0
        elif kind=='model_after':
            require(stage=='work' and position==48 and expect is None and not signal_trace and not program_trace and
                event==dict(kind='model_after',hashes=report['model_after']),'all48 before final frozen model guard')
            stage='sources'
        elif kind=='sources_final':
            require(stage=='sources' and event==dict(kind='sources_final',sha256=digest(p['sources'])),
                    'source closure after all actual work');stage='done'
        else:raise ValueError('unknown/duplicate/out-of-order interleaved journal record')
    require(stage=='done' and tail==report['journal_tail'] and gate.cursor==48,'complete gzip/hash chain/terminal48')
    require(canonical(gate.receipt())==canonical(report['session']['gate']),
            'entire v2 actual gate receipt, attempts/reservations/events/failure ticket preserved')
    aggregate={m:dict(attempted=budget.v1.zero(),completed=budget.v1.zero()) for m in contract.MODES}
    for row in report['session']['rows']:
        for receipt in (row['signal_resources'],None if row['compiler'] is None else row['compiler']['resource_receipt']):
            if receipt is None:continue
            core.audit_ledger(receipt)
            for kind in ('attempted','completed'):
                aggregate[row['mode']][kind]=budget.v1.add(aggregate[row['mode']][kind],receipt[kind])
    require(aggregate==gate.actual,'actual signal+compiler ledger equals predispatch gate on EVERY axis')
    return census,aggregate


def validate_report(output, report_sha, p, r):
    """Bounded actual artifacts and full native validators; no model execution."""
    core.keys(r,('schema','status','prepared','prepare_sha256','session','errors','model_before','model_after',
        'host','CUDA_peak_bytes','sources_verified','cleanup_ok','source_manifest_sha256','wall_seconds',
        'journal_tail','artifact_sha256',*FLAGS))
    require(r['schema']==NAME+'.report' and r['status']=='complete_native_panel_not_admitted' and r['errors']==[] and
        all(r[k] is False for k in FLAGS) and r['sources_verified'] is r['cleanup_ok'] is True and
        r['source_manifest_sha256']==digest(p['sources']),'complete terminal native report, closed science gates')
    session=r['session']
    core.keys(session,('schema','rows','gate','fixed_denominator','child_denominator','stopped','same_cap',
        'matched_actual_vector','native_verified',*FLAGS))
    require(session['schema']==native.NAME+'.ram_or_unverified_session' and len(session['rows'])==48 and
        session['fixed_denominator']==48 and session['child_denominator']==24 and session['stopped'] is False and
        session['same_cap'] is True and session['matched_actual_vector'] is session['native_verified'] is False and
        all(session[k] is False for k in FLAGS),'all48 full session, no cap-as-actual/native/scientific release')
    require({f.name for f in output.iterdir()}=={'report.json','binding.json',paid.JOURNAL} and
        all(f.is_file() and not f.is_symlink() for f in output.iterdir()) and
        set(r['artifact_sha256'])=={'binding.json',paid.JOURNAL},'exact exclusive artifact set')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'actual full artifact SHA')
    require(base.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),
            'actual execution prepare edge')
    pr=p['profile']
    require(sum(f.stat().st_size for f in output.iterdir())<=pr['output_bytes'] and
        (output/'report.json').stat().st_size<=pr['final_reserve_bytes'] and
        sum((output/n).stat().st_size for n in r['artifact_sha256'])<=pr['output_bytes']-pr['final_reserve_bytes'],
        'real encoded output and terminal partitions')
    require(r['model_before']==r['model_after'],'frozen before/after model and buffers')
    for hashes in (r['model_before'],r['model_after']):
        core.keys(hashes,('original_state_dict','all_buffers'))
        for h in hashes.values():core.hash_value(h)
    require(r['model_before']['original_state_dict']==p['expected_model_hash'],'actual original model identity')
    core.keys(r['host'],('device','index','name','allocation_sum','torch_version','torch_cuda','visible_nvidia_nodes'))
    require(r['host']['device']=='cuda:0' and type(r['host']['index']) is int and r['host']['index']==0 and
        r['host']['allocation_sum']==4. and all(type(r['host'][k]) is str and r['host'][k] for k in
        ('name','torch_version','torch_cuda')) and type(r['host']['visible_nvidia_nodes']) is list and
        r['host']['visible_nvidia_nodes'],'actual HOST metadata, not hardware attestation')
    require(type(r['CUDA_peak_bytes']) in (list,tuple) and len(r['CUDA_peak_bytes'])==2 and all(
        type(v) is int and 0<=v<=pr[k] for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),
        'actual CUDA peaks')
    require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<=pr['wall_seconds'],'bounded declared actual wall scope')
    census,actual=validate_stream(paid.iter_journal(output/paid.JOURNAL,
        decoded_cap=pr['decoded_journal_bytes'],line_cap=pr['journal_line_bytes']),p,r)
    contract.check_sources(p['sources']);pins()
    require(sha(output/'report.json')==report_sha and all(sha(output/n)==h for n,h in r['artifact_sha256'].items()),
            'final full report/artifact drift check')
    return dict(schema=NAME+'.consume',status='interleaved_native_fullarrays_budget_verified_not_admitted',
        report_sha256=report_sha,prepare_sha256=r['prepare_sha256'],fixed_denominator=48,child_denominator=24,
        independent_roots=2,census=census,actual_resources=actual,same_cap=True,
        actual_vectors_equal=actual['fixed']==actual['adaptive'],FLOPs=None,
        full_rows=session['rows'],gate_receipt_sha256=digest(session['gate']),
        verification_scope='saved full arrays, original chooser/ridge/replay and gate events; no neural/AD recomputation or GPU authenticity proof',
        saved_verification=dict(signal_array_SVD_calls=2*(census['compiled']+census['refused']),
            compiler_array_SVD_calls=census['compiled'],outside_native_ledger=True,neural_calls=0,physics_calls=0),**FLAGS)


def consume(output, expected):
    output=contract.absolute(output);core.hash_value(expected)
    require(sha(output/'report.json')==expected,'external actual terminal report SHA')
    r=base.read(output/'report.json',256*1024**2)
    p=verify(r['prepared'],r['prepare_sha256'])
    return validate_report(output,expected,p,r)


def exclusive(output):
    output=contract.absolute(output)
    require(output.parent==ROOT/'results' and output.name.startswith(NAME+'_') and not output.exists(),
            'NEW exclusive own result directory/file before verification/model access')
    return output


def write_new(output, value, cap=64*1024**2):
    output=exclusive(output);raw=canonical(value)
    require(len(raw)<=cap,'bounded complete saved metadata/receipt')
    require(shutil.disk_usage(output.parent).free>=profile()['disk_floor_bytes']+len(raw),'saved output disk floor')
    with output.open('xb') as f:
        require(f.write(raw)==len(raw),'complete exclusive metadata');f.flush();os.fsync(f.fileno())
    base.fsync_dir(output.parent)
    return dict(output=str(output),sha256=sha(output),bytes=len(raw))


def run(prepared, expected, output, authorization):
    require(authorization==AUTH,'MAIN authorization BEFORE verify, CUDA import or allocation')
    output=exclusive(output);started=time.monotonic();p=verify(prepared,expected)
    # Capacity refusal before model import. No quota/48-denominator reduction.
    require(time.monotonic()-started<=p['profile']['wall_seconds'],'verification exceeded whole scoped wall')
    require(shutil.disk_usage(output.parent).free>=p['profile']['disk_floor_bytes']+p['profile']['output_bytes'],
            'full512MiB reservation plus200MiB floor before native import')
    store=paid.Store(output,p['profile']);store.started=started
    model=None;before=after=host=None;peaks=None;cleanup=sources_ok=False;errors=[]
    active_slot=-1
    session=native.Session(p['wiring'],lambda e:store.event(dict(kind='work',slot=active_slot,payload=e)))
    try:
        store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=expected))
        store.event(dict(kind='denominator',rows=initial_rows(p),fixed_denominator=48,child_denominator=24,
                         sources_sha256=digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=base.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen state_dict')
        store.event(dict(kind='model_before',hashes=before))
        factories=native.NativeFactories(model,device,p['wiring']['tasks'],authorization=authorization)
        with sdpa_kernel(SDPBackend.MATH):
            for active_slot in range(48):
                store.check()
                session.next(factories,authorization=authorization)
                # next returns a copy before its finally block; the owned row is
                # authoritative and includes the finished signal ledger receipt.
                store.event(dict(kind='slot_return',slot=active_slot,row=deepcopy(session.rows[active_slot])))
        torch.cuda.synchronize(device);store.check();cleanup=True
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        if model is not None:
            try:
                after=base.model_hash(model)
                require(before==after and all(not m.training for m in model.modules()) and
                    all(not v.requires_grad for v in model.parameters()),'unchanged frozen native model/buffers')
                store.event(dict(kind='model_after',hashes=after))
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:
            contract.check_sources(p['sources']);pins();store.check();sources_ok=True
            store.event(dict(kind='sources_final',sha256=digest(p['sources'])))
        except BaseException as exc:errors.append(dict(stage='source_or_limits',message=str(exc)))
        try:store.close()
        except BaseException as exc:errors.append(dict(stage='journal_close',message=str(exc)))
        if store.memory is not None:peaks=list(store.memory())
        wall=time.monotonic()-started
        if wall>p['profile']['wall_seconds']:errors.append(dict(stage='wall',message='actual scoped wall exceeded'))
        if errors:
            # Preserve unknown denominator and failed partial work; do not retry,
            # relabel failed attempts as refusal, or overwrite the sealed source.
            for row in session.rows:
                if row['status']=='not_started':row['status']='blocked_not_dispatched'
        complete=not errors and cleanup and sources_ok and session.gate.cursor==48 and not session.stopped
        report=dict(schema=NAME+'.report',status='complete_native_panel_not_admitted' if complete else 'engineering_failed',
            prepared=str(prepared),prepare_sha256=expected,session=session.receipt(),errors=errors,
            model_before=before,model_after=after,host=host,CUDA_peak_bytes=peaks,sources_verified=sources_ok,
            cleanup_ok=cleanup,source_manifest_sha256=digest(p['sources']),wall_seconds=wall,
            journal_tail=store.tail,artifact_sha256={n:sha(output/n) for n in ('binding.json',paid.JOURNAL) if (output/n).exists()},**FLAGS)
        terminal_status=report['status'];terminal_errors=len(errors)
        try:store.put('report.json',report)
        except BaseException as exc:
            # Complete arrays remain in gzip. A bounded terminal failure record
            # keeps all48 identities and actual attempts; never admits partials.
            require(not (output/'report.json').exists(),'partial report write retained; no overwrite')
            minimal=dict(report);minimal['status']='engineering_failed'
            minimal['errors']=[*errors,dict(stage='final_report',type=type(exc).__name__,message=str(exc))]
            minimal['session']=dict(fixed_denominator=48,child_denominator=24,
                rows=[{k:r[k] for k in ('child_id','mode','status','error')} for r in session.rows],
                actual=session.gate.actual,pending=list(session.gate.pending.values()),
                full_arrays_retained_in_journal=True,full_receipt_missing=True)
            store.put('report.json',minimal)
            terminal_status='engineering_failed';terminal_errors=len(minimal['errors'])
    return dict(output=str(output),report_sha256=sha(output/'report.json'),
                status=terminal_status,errors=terminal_errors,fixed_denominator=48,child_denominator=24)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha256',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('verify');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha256',required=True)
    a=sub.add_parser('run');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha256',required=True);a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=sub.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha256',required=True);a.add_argument('--receipt',required=True)
    args=parser.parse_args()
    if args.cmd=='prepare':
        exclusive(args.output);result=write_new(args.output,material(args.prepared,args.prepare_sha256))
    elif args.cmd=='verify':
        verify(args.prepared,args.prepare_sha256);result=dict(status='strict_saved_host_prepare_verified',neural_calls=0,physics_calls=0)
    elif args.cmd=='run':result=run(args.prepared,args.prepare_sha256,args.output,args.authorization)
    else:
        exclusive(args.receipt);result=write_new(args.receipt,consume(args.output,args.report_sha256),profile()['saved_receipt_bytes'])
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
