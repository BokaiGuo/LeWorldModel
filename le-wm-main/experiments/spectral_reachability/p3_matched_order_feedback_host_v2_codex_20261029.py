"""Four-slot matched-order physical-feedback HOST CUDA runner.

Preparation and consumption never load a model. Actual run is explicitly
coordinated and remains known-root engineering evidence, never scientific pass.
"""
import argparse
import json
import os
from pathlib import Path
import time
from experiments.spectral_reachability import p3_matched_order_feedback_candidate_seal_codex_20261029 as candidate
from experiments.spectral_reachability import p3_matched_order_feedback_source_preflight_codex_20261029 as source
from experiments.spectral_reachability import p3_matched_order_feedback_lifecycle_codex_20261029 as life
from experiments.spectral_reachability import p3_matched_order_feedback_session_factory_v2_codex_20261029 as factory
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as storage

NAME='p3_matched_order_feedback_host_v2_codex_20261029'
AUTH='COORDINATED_MATCHED4_HOST_CUDA_PHYSICS'
ROOT=source.ROOT
CANDIDATE=ROOT/'results/p3_matched_order_feedback_candidate_seal_codex_20261029_MAIN01.json'
CANDIDATE_SHA='efb337ec58a353c9b9f90a94208813f0d9fb50b4760fadb72740866f0d26404d'
need=source.need
prior=life.prior


def build():
    old=source.host.design.read_pinned(CANDIDATE,CANDIDATE_SHA)
    need(old==candidate.build() and not old['runner_complete'],'candidate/source reconstruction')
    files=dict(old['implementation_sha256'])
    for module in (candidate,source,life,factory,factory.native,factory.native.encoder,storage,life.prior):
        path=Path(module.__file__).resolve();source.physical.merge(files,{str(path):source.host.sha(path)})
    for filename in (Path(__file__).resolve(),ROOT/'experiments/spectral_reachability/p3_matched_order_feedback_host_saved_v2_codex_20261029.py'):
        need(filename.is_file(),'own runner and independent consumer sources')
        source.physical.merge(files,{str(filename):source.host.sha(filename)})
    for name,h in files.items():need(Path(name).is_file() and not Path(name).is_symlink() and source.host.sha(name)==h,'implementation drift')
    return dict(schema=NAME+'.prepare',candidate_sha256=CANDIDATE_SHA,source_prepare_sha256=source.PREP_SHA,
        source_report_sha256=source.REPORT_SHA,source_receipt_sha256=source.RECEIPT_SHA,
        roster=old['roster'],implementation_sha256=files,limits=old['engineering_caps'],
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,precision=prior.precision_contract(),
        known_roots=2,fresh_roots=0,training=False,equal_FLOPs=False,scientific_pass=False,
        interpretation='known roots only; source-trusted host execution; no original P1/P3 gate change')


def verify(path,expected):
    value=source.host.design.read_pinned(Path(path),expected)
    need(value==build(),'prepare/source reconstruction')
    return value


def exclusive(path):
    p=Path(path).absolute();need(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists(),'exclusive own path')
    return p


def write_new(path,value):
    raw=source.host.core.canonical(value)
    with Path(path).open('xb') as f:
        need(f.write(raw)==len(raw),'full output write');f.flush();os.fsync(f.fileno())
    source.host.signal.fsync_dir(Path(path).parent)
    return source.host.sha(path)


def prepare(path):
    out=exclusive(path);return write_new(out,build())


def hook_snapshot(model):
    return [(m,'forward' in m.__dict__,m.__dict__.get('forward'),
             tuple(m._forward_hooks.items()),tuple(m._forward_pre_hooks.items()),
             tuple(m._backward_hooks.items())) for m in model.modules()]


def hooks_restored(snapshot):
    return all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is old and
               tuple(m._forward_hooks.items())==post and tuple(m._forward_pre_hooks.items())==pre and
               tuple(m._backward_hooks.items())==back for m,had,old,post,pre,back in snapshot)


def run(prepared,prepare_sha,output,authorization,coordination):
    need(authorization==AUTH and type(coordination) is str and len(coordination.strip())>=8,
         'explicit HOST authorization and GPU coordination record required')
    start=time.monotonic();p=verify(prepared,prepare_sha);original=source.host.verify_prepare(source.PREP,source.PREP_SHA)
    data=source.physical.source_material();out=exclusive(output);store=storage.Store(out)
    rows=[dict(slot=s,root=r,policy=q,status='not_started',result_sha256=None,cleanup='not_required',error=None)
          for s,r,q in life.seam.ROSTER]
    errors=[];model=None;before=after=classes=classes_after=gpu=None;precision=[];source_ok=False; peak=None;hooks=None;hooks_ok=False
    try:
        store.event('physical','record','matched_run_start',store.index,
                    dict(prepare_sha256=prepare_sha,coordination=coordination,roster=p['roster']))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.checkpoint_compat import load_compatible_pretrained
        gpu=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def guard():
            store.check('numerical_events',0);prior.checked_precision(torch)
            need(torch.cuda.max_memory_allocated(device)<=p['cuda_allocated_bytes'] and
                 torch.cuda.max_memory_reserved(device)<=p['cuda_reserved_bytes'],'whole-job CUDA memory bounds')
        model=load_compatible_pretrained(original['policy'],device=device).eval().requires_grad_(False)
        before=source.host.signal.model_hash(model);need(before['original_state_dict']==original['expected_model_hash'],'model identity')
        classes=prior.model_sources(model,original)
        hooks=hook_snapshot(model)
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            precision.append(prior.checked_precision(torch));guard()
            make=factory.Factory(original,data,store,model,torch,device)
            def backend(task,sink):
                return prior.RecordingBackend(prior.NativeTree(model,device,task,sink),task,sink,guard)
            result=life.execute(original['tasks'],store,make,backend);rows=result['rows']
            precision.append(prior.checked_precision(torch));guard()
        torch.cuda.synchronize(device)
        need(hooks_restored(hooks),'restored forwards and hooks')
        peak=[int(torch.cuda.max_memory_allocated(device)),int(torch.cuda.max_memory_reserved(device))]
    except BaseException as exc:
        errors.append(dict(stage='run',type=type(exc).__name__,message=str(exc)))
    finally:
        if model is not None:
            try:
                hooks_ok=hooks is not None and hooks_restored(hooks)
                need(hooks_ok,'hook/forward cleanup after success or exception')
                after=source.host.signal.model_hash(model);classes_after=prior.model_sources(model,original)
                need(after==before and classes_after==classes and all(not m.training for m in model.modules()) and
                     all(not q.requires_grad for q in model.parameters()),'final frozen model/classes')
            except BaseException as exc:errors.append(dict(stage='model_after',type=type(exc).__name__,message=str(exc)))
        try:
            verify(prepared,prepare_sha);source_ok=True
        except BaseException as exc:errors.append(dict(stage='final_source',type=type(exc).__name__,message=str(exc)))
        if not 0 <= time.monotonic()-start <= storage.WALL:
            errors.append(dict(stage='wall',type='DeadlineExceeded',message='whole job wall cap'))
        # Recover actually settled identities if an outer fault interrupted the lifecycle.
        journal=out/'events.jsonl'
        if journal.is_file():
            try:
                events=storage.inspect_journal(journal,source.host.sha(journal))['events']
                settled=[e['detail'] for e in events if e['origin']=='physical' and e['resource']=='matched_slot_return']
                for row in settled:
                    need(type(row['slot']) is int and 0<=row['slot']<4,'settled slot');rows[row['slot']]=row
            except BaseException as exc:errors.append(dict(stage='recovery',type=type(exc).__name__,message=str(exc)))
        complete=not errors and source_ok and all(r['status']=='callback_feedback_complete_not_certified' and
                 r['cleanup']=='returned' for r in rows) and not store.pending and not store.failures
        report=dict(schema=NAME+'.report',status='four_complete_candidate' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,coordination=coordination,rows=rows,errors=errors,
            model_before=before,model_after=after,classes_before=classes,classes_after=classes_after,precision=precision,
            gpu=gpu,peak_bytes=peak,sources_verified=source_ok,hooks_restored=hooks_ok,elapsed_seconds=time.monotonic()-start,
            journal_sha256=source.host.sha(journal) if journal.is_file() else None,journal_tail=store.previous,
            event_count=store.index,category_bytes=store.used,pending=[dict(origin=k[0],token=k[1],resource=v) for k,v in sorted(store.pending.items())],
            known_roots=2,fresh_roots=0,training=False,equal_FLOPs=False,scientific_pass=False)
        # Final-reserve writes are exclusive and deliberately independent of normal category limits.
        report_path=out/'report.json';report_sha=write_new(report_path,report)
        completion=dict(schema=NAME+'.completion',report_sha256=report_sha,complete=complete,
                        boundary='after_report_fsync',scientific_pass=False)
        completion_sha=write_new(out/'completion.json',completion)
    return dict(output=str(out),report_sha256=report_sha,completion_sha256=completion_sha,status=report['status'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha256',required=True)
    a.add_argument('--output',required=True);a.add_argument('--authorization',required=True);a.add_argument('--gpu-coordination',required=True)
    a=sub.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha256',required=True)
    a.add_argument('--completion-sha256',required=True);a.add_argument('--receipt',required=True)
    a=ap.parse_args()
    if a.command=='prepare':result=dict(prepare_sha256=prepare(a.output))
    elif a.command=='run':result=run(a.prepared,a.prepare_sha256,a.output,a.authorization,a.gpu_coordination)
    else:
        from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as consumer
        result=dict(receipt_sha256=write_new(exclusive(a.receipt),consumer.consume(a.output,a.report_sha256,a.completion_sha256)))
    print(json.dumps(result,sort_keys=True))
    if a.command=='run' and result['status']!='four_complete_candidate':raise SystemExit(2)


if __name__=='__main__':main()
