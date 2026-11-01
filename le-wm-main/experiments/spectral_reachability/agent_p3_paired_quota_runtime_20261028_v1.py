"""New paired predict-quota CUDA experiment. Saved/RAM only unless MAIN authorizes run.

Each child pays independent fixed/adaptive H1+H3 signals, pairseals, then really
compiles H1x12/H3x4 (24 search+1 selected-prefix). Never mutates original decision.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from experiments.spectral_reachability import agent_p3_child_budget_host_cuda_20261028_v1 as host
from experiments.spectral_reachability import agent_p3_paired_quota_runtime_20261028_v1_gate as pair
from experiments.spectral_reachability import agent_p3_paired_quota_runtime_20261028_v1_consumer as consumer

NAME='agent_p3_paired_quota_runtime_20261028_v1'
ROOT,core,contract,paid,base=host.ROOT,host.core,host.contract,host.paid,host.base
native,compiler=host.native,pair.compiler
require,sha,digest,canonical=host.require,host.sha,host.digest,host.canonical
FLAGS=dict(host.FLAGS)
AUTH=pair.AUTH
UPSTREAM=ROOT/'results/agent_p3_child_budget_host_cuda_20261028_v1_prepare_MAIN01.json'
UPSTREAM_SHA='92f2edacae8e0f519e09d573ed1c78c61247e78b7b99042e27e29f39868146c8'
initial_rows=pair.initial_rows
validate_stream=consumer.validate_stream

def profile():
    p=host.profile()
    p.update(decoded_journal_bytes=2*1024**3,slot_trace_bytes=64*1024**2,
        capacity_forecast='no asserted fit; H1 12 candidates; actual512MiB output/256MiB journal partition fails closed',
        matched_units=['predict_API','predict_samples'],equal_FLOPs=False)
    return p

def own_files():
    return [Path(__file__).resolve(),Path(compiler.__file__).resolve(),Path(pair.__file__).resolve(),
        Path(consumer.__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']

def pins():
    host.pins()
    return {str(Path(host.__file__).resolve()):'a0dbd2bcae92fc10e541500ee3b8247ec72a7e5c7a79c67f0248761174354b14'}

def material(prepared,expected):
    require(str(contract.absolute(prepared))==str(UPSTREAM) and expected==UPSTREAM_SHA,'fixed actual original source prepare')
    contract.check_sources(pins())
    original=host.verify(prepared,expected)
    children=original['wiring']['children'];tasks=original['wiring']['tasks']
    wiring=dict(children=deepcopy(children),tasks=deepcopy(tasks),limits=pair.limits())
    pair.Gate(wiring)
    sources=deepcopy(original['sources'])
    contract.merge(sources,pins())
    contract.merge(sources,{str(f):sha(f) for f in own_files()})
    contract.merge(sources,{str(UPSTREAM):expected})
    contract.check_sources(sources)
    require(sha(UPSTREAM)==expected,'original prepare unchanged')
    return dict(schema=NAME+'.prepare',native_prepared=str(prepared),native_sha256=expected,
        wiring=wiring,policy=original['policy'],expected_model_hash=original['expected_model_hash'],
        sources=sources,profile=profile(),fixed_denominator=48,child_denominator=24,independent_roots=2,
        decision_source='unchanged original chooser; independently paid NEW signals',
        execution_quota='H1=12,H3=4; below original requested count refuses BOTH after signal payment',
        cap_vector_sha256=digest(pair.limits()),signal_sharing='independent',scientific_equal_actual=False,**FLAGS)

def verify(prepared,expected):
    path=contract.absolute(prepared);core.hash_value(expected)
    require(sha(path)==expected,'external actual paired prepare SHA')
    p=base.read(path,64*1024**2);contract.check_sources(p['sources'])
    require(canonical(p)==canonical(material(p['native_prepared'],p['native_sha256'])) and sha(path)==expected,
        'complete original source/context/policy and NEW quota/profile rebuild')
    return p

def validate_report(output, report_sha, p, r):
    """Bounded actual artifacts and full native validators; no model execution."""
    core.keys(r,('schema','status','prepared','prepare_sha256','session','errors','model_before','model_after',
        'host','CUDA_peak_bytes','sources_verified','cleanup_ok','source_manifest_sha256','wall_seconds',
        'journal_tail','artifact_sha256',*FLAGS))
    require(r['schema']==NAME+'.report' and r['status']=='complete_native_panel_not_admitted' and r['errors']==[] and
        all(r[k] is False for k in FLAGS) and r['sources_verified'] is r['cleanup_ok'] is True and
        r['source_manifest_sha256']==digest(p['sources']),'complete terminal native report, closed science gates')
    session=r['session']
    core.keys(session,('schema','rows','gate','fixed_denominator','child_denominator','stopped'))
    require(session['schema']==NAME+'.session' and len(session['rows'])==48 and
        session['fixed_denominator']==48 and session['child_denominator']==24 and session['stopped'] is False,
        'complete new paired24/48 session')
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
    return dict(schema=NAME+'.consume',status='paired_predict_quota_fullarrays_verified_not_admitted',
        report_sha256=report_sha,prepare_sha256=r['prepare_sha256'],fixed_denominator=48,child_denominator=24,
        independent_roots=2,census=census,actual_resources=actual,same_cap=True,
        actual_vectors_equal=actual['fixed']==actual['adaptive'],matched_predict_API_samples=True,FLOPs=None,
        full_rows=session['rows'],gate_receipt_sha256=digest(session['gate']),
        verification_scope='saved full arrays, original chooser/ridge/replay and gate events; no neural/AD recomputation or GPU authenticity proof',
        saved_verification=dict(signal_array_SVD_calls=2*(census['compiled']+census['paired_paid_refusal']),
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
    model=None;before=after=host_metadata=None;peaks=None;cleanup=sources_ok=False;errors=[]
    active_slot=-1
    session=pair.Session(p,store.event)
    try:
        store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=expected))
        store.event(dict(kind='denominator',rows=initial_rows(p),fixed_denominator=48,child_denominator=24,
                         sources_sha256=digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host_metadata=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=base.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen state_dict')
        for module in model.modules():
            path=str(Path(inspect.getfile(type(module))).resolve())
            require(path in p['sources'] and sha(path)==p['sources'][path],'bound loaded native class source')
        store.event(dict(kind='model_before',hashes=before))
        factories=native.NativeFactories(model,device,p['wiring']['tasks'],authorization=host.AUTH)
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for active_slot in range(24):
                store.check()
                session.next_pair(factories,authorization=authorization)
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
        complete=not errors and cleanup and sources_ok and session.gate.cursor==24 and not session.stopped
        report=dict(schema=NAME+'.report',status='complete_native_panel_not_admitted' if complete else 'engineering_failed',
            prepared=str(prepared),prepare_sha256=expected,session=session.receipt(),errors=errors,
            model_before=before,model_after=after,host=host_metadata,CUDA_peak_bytes=peaks,sources_verified=sources_ok,
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
                recorded_arrays_retained_in_journal=True,full_receipt_missing=True)
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
        verify(args.prepared,args.prepare_sha256);result=dict(status='strict_saved_paired_prepare_verified',neural_calls=0,physics_calls=0)
    elif args.cmd=='run':result=run(args.prepared,args.prepare_sha256,args.output,args.authorization)
    else:
        exclusive(args.receipt);result=write_new(args.receipt,consume(args.output,args.report_sha256),profile()['saved_receipt_bytes'])
    print(json.dumps(result,sort_keys=True))
    return 1 if args.cmd=='run' and result['status']!='complete_native_panel_not_admitted' else 0


if __name__=='__main__':raise SystemExit(main())

