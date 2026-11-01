"""Source-first four-slot E1 host runner; MAIN review required before CUDA."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_e1_same_target_durable_staging_20261029_v1 as stage
from experiments.spectral_reachability import agent_e1_same_target_source_meter_20261029_v1 as source

old = stage.old
ROOT = source.ROOT
NAME = 'agent_e1_same_target_host_cuda_20261029_v1'
CAP = old.CAP
WALL = old.WALL
GPU_CAP = 4 * 1024**3
LOCAL = ('agent_e1_same_target_host_cuda_20261029_v1',
         'agent_e1_same_target_durable_staging_20261029_v1',
         'agent_e1_same_target_four_slot_execution_20261029_v1',
         'agent_e1_same_target_backend_pair_20261029_v1',
         'agent_e1_same_target_source_meter_20261029_v1',
         'agent_e1_same_target_candidate_intervention_20261029_v1',
         'agent_e1_six_slot_common_cuda_20261029_v2',
         'agent_e1_six_slot_controller_core_20261029_v1',
         'agent_e1_six_slot_phase_meter_20261029_v1')


def stream_sha(path):
    p = Path(path)
    old.require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'absolute regular source')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def authorities():
    a = source.source_inputs()
    p,h = source.prep.FILES['controller_prepare']
    previous = source.prep.parse(source.prep.bytes_at(ROOT/p,h))
    return a, previous


def source_closure(previous):
    paths = [ROOT/'experiments/spectral_reachability'/(x+'.py') for x in LOCAL]
    paths += [ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    current = {str(p):stream_sha(p) for p in paths}
    # Preserve the complete original declared closure, including checkpoint data.
    for path, expected in previous['bindings_sha256'].items():
        old.require(stream_sha(path) == expected, 'original source/model binding drift: '+path)
        if path in current: old.require(current[path] == expected, 'overlapping source disagreement')
        current[path] = expected
    return current


def manifest():
    authority, previous = authorities()
    return dict(schema=NAME+'.prepare', authority=authority,
                closure=source_closure(previous), expected_model_hash=previous['expected_model_hash'],
                storage=old.storage_bound(), slots=[list(x) for x in stage.four.SLOTS],
                per_phase=dict(controller=144,extra=2,replay=4), per_axis_total=600,
                cuda_only=True, gpu_allocator_cap=GPU_CAP, wall=WALL,
                scientific_admission=False, physical_reachability_authenticated=False,
                capacity_note='enforced caps, not proof that native events fit')


def verify_manifest(path, expected):
    saved = old.parse_canonical(old.read_bound(Path(path),expected,16*1024*1024))
    old.require(saved == manifest(), 'manifest/source/authority drift')
    return saved


def tasks_and_targets(authority, previous):
    tasks=[]; targets=[]; prefixes=[]
    for i,row in enumerate(authority['rows']):
        slot=previous['slots'][i]
        task=old.core.old.sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        old.require(old.core.old.core.task_id(task)==row['task_id']==slot['task_id'], 'actual task identity')
        target=np.frombuffer(bytes.fromhex(row['target_hex']),dtype='<f4')
        old.require(target.shape==(192,) and np.isfinite(target).all(), 'target bytes ABI')
        tasks.append(task);targets.append(target.astype(float).tolist());prefixes.append(row['witness_raw5'])
    return tasks,targets,prefixes


def run(prepared, expected, output):
    started=time.monotonic(); binding=verify_manifest(prepared,expected)
    authority,previous=authorities(); tasks,targets,prefixes=tasks_and_targets(authority,previous)
    store=old.Store(Path(output),started=started)
    staged=dict(status='partial_or_failed', rows=[dict(root=r,arm=a,status='not_started') for r,a in stage.four.SLOTS],
                errors=[], result_sealed=False, authority_authenticated=False,scientific_admission=False)
    model=None; before=after=loaded=host=gpu=None; torch=None
    try:
        store.event(dict(kind='authority_sealed_before_model',sha256=old.sha(old.canonical(authority))))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch as torch_module
        torch=torch_module
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from hydra.utils import instantiate
        from experiments.checkpoint_compat import remap_legacy_vit_keys
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats(device)
        policy=Path(previous['policy']); config=policy/'config.json';weights=policy/'weights.pt'
        config_bytes=old.read_bound(config,previous['bindings_sha256'][str(config)],4*1024*1024)
        weight_bytes=old.read_bound(weights,previous['bindings_sha256'][str(weights)],512*1024*1024)
        model,loaded=old.load_bound_buffers(config_bytes,weight_bytes,instantiate,
            lambda stream:torch.load(stream,map_location='cpu',weights_only=True),remap_legacy_vit_keys,device)
        del config_bytes,weight_bytes
        old.core.old.enc.frozen(model)
        old.require(all(p.device.type=='cuda' for p in model.parameters()),'CUDA only')
        before=old.core.old.enc.tensor_hash(model)
        old.require(before==binding['expected_model_hash'],'frozen loaded model')
        store.event(dict(kind='model_loaded',loaded=loaded,model_hash=before,cuda=True,
                         tf32=False,autocast=False,math_requested=True))
        def check():
            store.check()
            old.require(not torch.is_autocast_enabled('cuda') and not torch.backends.cuda.matmul.allow_tf32
                        and not torch.backends.cudnn.allow_tf32, 'precision guard')
            old.require(torch.cuda.max_memory_allocated(device)<=GPU_CAP and torch.cuda.max_memory_reserved(device)<=GPU_CAP,'GPU allocator cap')
        backends=[old.phase.beam.NativeEdge(model,device,tasks[r]) for r,_ in stage.four.SLOTS]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            staged=stage.execute(store,tasks,targets,prefixes,lambda i:backends[i],
                                 lambda i,meter:backends[i].guard(meter),lambda:torch.cuda.synchronize(device),check)
        check()
    except BaseException as exc:
        staged['status']='partial_or_failed';staged['errors'].append(dict(type=type(exc).__name__,message=str(exc)[:2048]))
    finally:
        if model is not None:
            try:
                torch.cuda.synchronize(device);old.core.old.enc.frozen(model);after=old.core.old.enc.tensor_hash(model)
                gpu=dict(peak_allocated=int(torch.cuda.max_memory_allocated(device)),peak_reserved=int(torch.cuda.max_memory_reserved(device)),
                         cap=GPU_CAP,tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                         tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),autocast=bool(torch.is_autocast_enabled('cuda')),
                         requested_math=True,observed_math=None)
                old.require(before==after==binding['expected_model_hash'],'frozen model after')
            except BaseException as exc:
                staged['status']='partial_or_failed';staged['errors'].append(dict(type=type(exc).__name__,message='final model: '+str(exc)[:2048]))
        try: verify_manifest(prepared,expected)
        except BaseException as exc:
            staged['status']='partial_or_failed';staged['errors'].append(dict(type=type(exc).__name__,message='final closure: '+str(exc)[:2048]))
        staged.update(schema=NAME+'.report',prepared=str(Path(prepared)),prepare_sha256=expected,
                      model_hash_before=before,model_hash_after=after,loaded=loaded,host=host,gpu=gpu,
                      storage=old.storage_bound(), training=False,new_physics=0,new_encoder=0,new_ad=0)
        terminal=stage.finish(store,staged)
    old.require(terminal['report']['status']=='all_four_sealed','partial evidence preserved; no retry')
    timing=old.parse_canonical((store.path/'timing.json').read_bytes())
    old.require(timing['wall_within_limit'] is True,'wall exceeded; evidence preserved')
    return {k:v for k,v in terminal.items() if k!='report'}


def consume(output, report_sha, completion_sha, timing_sha):
    output=Path(output)
    old.require(output.is_absolute() and output.is_dir() and not output.is_symlink(),'output directory')
    report=old.parse_canonical(old.read_bound(output/'report.json',report_sha,old.RESERVE))
    completion=old.parse_canonical(old.read_bound(output/'completion.json',completion_sha,old.RESERVE))
    timing=old.parse_canonical(old.read_bound(output/'timing.json',timing_sha,old.RESERVE))
    binding=verify_manifest(report['prepared'],report['prepare_sha256'])
    authority,previous=authorities();tasks,targets,prefixes=tasks_and_targets(authority,previous)
    result=consume_authenticated(output,report_sha,completion_sha,timing_sha,report,completion,timing,
                                 binding,authority,previous,tasks,targets,prefixes)
    verify_manifest(report['prepared'],report['prepare_sha256'])
    return result


def consume_authenticated(output, report_sha, completion_sha, timing_sha, report, completion, timing,
                          binding, authority, previous, tasks, targets, prefixes):
    """Internal saved checker; callers must authenticate all supplied objects."""
    report_keys={'status','rows','errors','result_sealed','authority_authenticated','scientific_admission',
                 'schema','prepared','prepare_sha256','model_hash_before','model_hash_after','loaded','host','gpu',
                 'storage','training','new_physics','new_encoder','new_ad','artifacts','journal_events',
                 'journal_sha256','array_bytes','metadata_bytes'}
    old.require(set(report)==report_keys and report['schema']==NAME+'.report' and report['errors']==[] and report['status']=='all_four_sealed'
                and report['training'] is False and report['new_physics']==report['new_encoder']==report['new_ad']==0
                and report['scientific_admission'] is False and report['storage']==old.storage_bound()
                and report['model_hash_before']==report['model_hash_after']==binding['expected_model_hash'], 'terminal status/model')
    old.require(completion==dict(schema=old.NAME+'.completion',report_sha256=report_sha,
                timing_boundary='completion file and directory fsync measured by separate timing witness')
                and set(timing)=={'schema','report_sha256','completion_sha256','elapsed_after_completion_fsync',
                                  'wall_limit','wall_within_limit','timing_boundary'}
                and timing['schema']==old.NAME+'.timing' and timing['report_sha256']==report_sha
                and timing['completion_sha256']==completion_sha and timing['wall_limit']==WALL
                and timing['wall_within_limit'] is True and type(timing['elapsed_after_completion_fsync']) in (int,float)
                and timing['timing_boundary']=='run entry through completion file and directory fsync; excludes timing witness fsync'
                and 0<=timing['elapsed_after_completion_fsync']<=WALL, 'completion/wall')
    gpu=report['gpu'];old.require(gpu['cap']==GPU_CAP and 0<=gpu['peak_allocated']<=GPU_CAP
                and 0<=gpu['peak_reserved']<=GPU_CAP and gpu['tf32_matmul'] is gpu['tf32_cudnn'] is gpu['autocast'] is False
                and gpu['requested_math'] is True and gpu['observed_math'] is None,'GPU precision/cap')
    policy=Path(previous['policy']); loaded=report['loaded']
    old.require(loaded['config_sha256']==previous['bindings_sha256'][str(policy/'config.json')]
                and loaded['weights_sha256']==previous['bindings_sha256'][str(policy/'weights.pt')]
                and loaded['strict_key_coverage'] is True,'loaded bytes')
    old.require(sum(p.stat().st_size for p in output.iterdir())<=CAP,'total cap')
    payload,events=old.read_store(output,report)
    old.require(len(events)>=2 and events[0]==dict(kind='authority_sealed_before_model',sha256=old.sha(old.canonical(authority)))
                and events[1]==dict(kind='model_loaded',loaded=loaded,model_hash=binding['expected_model_hash'],
                                    cuda=True,tf32=False,autocast=False,math_requested=True), 'authority/model event ordering')
    result=stage.verify_payload(tasks,targets,prefixes,report,payload,events[2:])
    return dict(schema=NAME+'.consume',status='four_saved_controller_replay_verified',report_sha256=report_sha,
                completion_sha256=completion_sha,timing_sha256=timing_sha,verification=result,
                per_axis=600,scientific_admission=False,physical_reachability_authenticated=False,
                boundary='saved module IO/loader attestation; no neural re-execution')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    parser.add_argument('--output');parser.add_argument('--prepared');parser.add_argument('--prepare-sha')
    parser.add_argument('--report-sha');parser.add_argument('--completion-sha');parser.add_argument('--timing-sha');parser.add_argument('--receipt')
    a=parser.parse_args()
    if a.command=='prepare':
        p=Path(a.output);old.require(p.is_absolute(),'absolute output');out=dict(path=str(p),sha256=old.put_new(p.parent,p.name,old.canonical(manifest())))
    elif a.command=='verify':
        verify_manifest(a.prepared,a.prepare_sha);out=dict(status='prepare_verified')
    elif a.command=='run': out=run(a.prepared,a.prepare_sha,a.output)
    else:
        receipt=consume(a.output,a.report_sha,a.completion_sha,a.timing_sha)
        p=Path(a.receipt);old.require(p.is_absolute(),'absolute receipt');out=dict(path=str(p),sha256=old.put_new(p.parent,p.name,old.canonical(receipt)))
    print(json.dumps(out,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
